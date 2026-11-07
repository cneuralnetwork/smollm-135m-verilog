// Behavioral SystemVerilog inference for HuggingFaceTB/SmolLM-135M.
module smollm;
  localparam int H=576, F=1536, V=49152, L=30, M=48900, S=2048;
  localparam int WMAX=F*H, BMAX=WMAX*4, VB=332367, U_RANGES=1480;
  byte unsigned raw[0:BMAX-1];
  real w[0:WMAX-1], emb[0:V*H-1];
  real x[0:H-1], a[0:H-1], q[0:H-1], k[0:191], val[0:191];
  real att[0:H-1], up[0:F-1], gate[0:F-1], score[0:S-1];
  real kc[0:L-1][0:S-1][0:191], vc[0:L-1][0:S-1][0:191];
  real pow2[0:255], inv[0:31];
  logic [63:0] off[0:271];
  logic [15:0] byte_id[0:255], ml[0:M-1], mr[0:M-1], mo[0:M-1], mk[0:M-1];
  logic [23:0] vst[0:V-1];
  logic [15:0] vlen[0:V-1];
  logic [7:0] vb[0:VB-1];
  logic [20:0] ustart[0:U_RANGES-1], uend[0:U_RANGES-1];
  logic [1:0] ukind[0:U_RANGES-1];
  int toks[0:S-1], piece[0:8191];
  int fd, got, ntok, np, next_id, max_new;
  bit machine_mode;
  string prompt, model_path;
  real norm,best;

  function automatic real f32(input int unsigned bits);
    int e,m;
    real z;
    e=(bits>>23)&255;m=bits&32'h7fffff;
    if(e==0) z=(m/8388608.0)*pow2[1];
    else if(e==255) z=0.0;
    else z=(1.0+m/8388608.0)*pow2[e];
    return bits[31] ? -z:z;
  endfunction
  function automatic int unsigned word_at(input int i);
    return int'(raw[i]) | (int'(raw[i+1])<<8) |
           (int'(raw[i+2])<<16) | (int'(raw[i+3])<<24);
  endfunction
  task automatic read_at(input longint offset,input int count);
    if($fseek(fd,int'(offset),0)!=0) $fatal(1,"seek failed: %0d",offset);
    got=$fread(raw,fd,0,count*4);
    if(got!=count*4) $fatal(1,"short read %0d/%0d",got,count*4);
  endtask
  task automatic load(input int tensor,input int count);
    read_at(off[tensor],count);
    for(int i=0;i<count;i++) w[i]=f32(word_at(i*4));
  endtask
  // dst: q/k/v/att/gate/up or residual addition.
  task automatic matvec(input int rows,input int cols,input int dst);
    real z;
    for(int i=0;i<rows;i++) begin
      z=0.0;
      for(int j=0;j<cols;j++) z+=w[i*cols+j]*a[j];
      case(dst)
        0:q[i]=z; 1:k[i]=z; 2:val[i]=z; 3:att[i]=z;
        4:gate[i]=z; 5:up[i]=z; 6:x[i]+=z;
      endcase
    end
  endtask
  task automatic rms(input int tensor);
    load(tensor,H);norm=0.0;
    for(int i=0;i<H;i++) norm+=x[i]*x[i];
    norm=1.0/$sqrt(norm/H+1.0e-5);
    for(int i=0;i<H;i++) a[i]=x[i]*norm*w[i];
  endtask
  function automatic int find_merge(input int left,input int right);
    int lo,hi,mid,cmp;
    lo=0;hi=M-1;
    while(lo<=hi) begin
      mid=(lo+hi)/2;cmp=int'(ml[mid])-left;
      if(cmp==0) cmp=int'(mr[mid])-right;
      if(cmp<0) lo=mid+1;
      else if(cmp>0) hi=mid-1;
      else return mid;
    end
    return -1;
  endfunction
  task automatic encode(input int start,input int stop);
    int r,rank,at,mi;
    np=0;
    for(int i=start;i<stop;i++) begin
      if(byte_id[int'(prompt[i])]==16'hffff)
        $fatal(1,"unsupported prompt byte %0d",int'(prompt[i]));
      if(np>=8192) $fatal(1,"one prompt segment exceeds 8192 bytes");
      piece[np]=int'(byte_id[int'(prompt[i])]);np++;
    end
    while(np>1) begin
      rank=M+1;at=-1;mi=-1;
      for(int i=0;i<np-1;i++) begin
        r=find_merge(piece[i],piece[i+1]);
        if(r>=0 && int'(mk[r])<rank) begin rank=int'(mk[r]);at=i;mi=r;end
      end
      if(at<0) break;
      piece[at]=int'(mo[mi]);
      for(int i=at+1;i<np-1;i++) piece[i]=piece[i+1];
      np--;
    end
    for(int i=0;i<np;i++) begin
      if(ntok>=S) $fatal(1,"prompt exceeds %0d tokens",S);
      toks[ntok]=piece[i];ntok++;
    end
  endtask
  function automatic string special_text(input int id);
    case(id)
      0:return "<|endoftext|>"; 1:return "<|im_start|>";
      2:return "<|im_end|>"; 3:return "<repo_name>";
      4:return "<reponame>"; 5:return "<file_sep>";
      6:return "<filename>"; 7:return "<gh_stars>";
      8:return "<issue_start>"; 9:return "<issue_comment>";
      10:return "<issue_closed>"; 11:return "<jupyter_start>";
      12:return "<jupyter_text>"; 13:return "<jupyter_code>";
      14:return "<jupyter_output>"; 15:return "<jupyter_script>";
      16:return "<empty_output>";
      default:return "";
    endcase
  endfunction
  function automatic int special_at(input int at);
    string s;
    bit ok;
    for(int id=0;id<17;id++) begin
      s=special_text(id);
      if(at+s.len()<=prompt.len()) begin
        ok=1;
        for(int j=0;j<s.len();j++) if(prompt[at+j]!=s[j]) ok=0;
        if(ok) return id;
      end
    end
    return -1;
  endfunction
  function automatic int utf8_step(input int at);
    int c;
    c=int'(prompt[at]);
    if(c<128) return 1;
    if((c&32'hE0)==32'hC0 && at+1<prompt.len()) return 2;
    if((c&32'hF0)==32'hE0 && at+2<prompt.len()) return 3;
    if((c&32'hF8)==32'hF0 && at+3<prompt.len()) return 4;
    return 1;
  endfunction
  function automatic int codepoint_at(input int at);
    int c,step;
    c=int'(prompt[at]);step=utf8_step(at);
    case(step)
      2:return ((c&31)<<6)|(int'(prompt[at+1])&63);
      3:return ((c&15)<<12)|((int'(prompt[at+1])&63)<<6)|(int'(prompt[at+2])&63);
      4:return ((c&7)<<18)|((int'(prompt[at+1])&63)<<12)|
               ((int'(prompt[at+2])&63)<<6)|(int'(prompt[at+3])&63);
      default:return c;
    endcase
  endfunction
  function automatic int cls_at(input int at);
    int cp,lo,hi,mid;
    cp=codepoint_at(at);
    if(cp<0||cp>1114111) return 3;
    lo=0;hi=U_RANGES-1;
    while(lo<=hi) begin
      mid=(lo+hi)/2;
      if(cp<int'(ustart[mid])) hi=mid-1;
      else if(cp>int'(uend[mid])) lo=mid+1;
      else return int'(ukind[mid]);
    end
    return 3;
  endfunction
  task automatic tokenize;
    int i,start,kind,j,last,ch,c1,c2,sp;
    ntok=0;i=0;
    if($test$plusargs("bos")) begin toks[0]=0;ntok=1;end
    while(i<prompt.len()) begin
      sp=special_at(i);
      if(sp>=0) begin
        if(ntok>=S) $fatal(1,"prompt exceeds %0d tokens",S);
        toks[ntok]=sp;ntok++;i+=special_text(sp).len();continue;
      end
      start=i;ch=int'(prompt[i]);kind=cls_at(i);
      if(kind==0) begin
        j=i;last=i;
        while(j<prompt.len() && cls_at(j)==0) begin last=j;j+=utf8_step(j);end
        if(last>i && j<prompt.len()) begin
          // ByteLevel leaves the last whitespace character for the next match.
          encode(i,last);i=last;continue;
        end
        if(ch==32 && i+1<prompt.len() && cls_at(i+1)!=0) begin
          i++;kind=cls_at(i);
        end
      end
      if(i==start && ch==39 && i+1<prompt.len()) begin
        c1=int'(prompt[i+1]);
        if(c1>=65 && c1<=90) c1+=32;
        c2=(i+2<prompt.len()) ? int'(prompt[i+2]):0;
        if(c2>=65 && c2<=90) c2+=32;
        if(c1==115 || c1==116 || c1==109 || c1==100) begin
          i+=2;encode(start,i);continue;
        end
        if((c1==114&&c2==101)||(c1==118&&c2==101)||
           (c1==108&&c2==108)) begin
          i+=3;encode(start,i);continue;
        end
      end
      i+=utf8_step(i);
      if(kind!=2) while(i<prompt.len() && cls_at(i)==kind && special_at(i)<0) i+=utf8_step(i);
      encode(start,i);
    end
    if(ntok==0) begin toks[0]=0;ntok=1;end
  endtask
  task automatic emit(input int id);
    for(int i=0;i<int'(vlen[id]);i++) $write("%c",vb[int'(vst[id])+i]);
    $fflush();
  endtask
  task automatic emit_machine(input int id);
    $write("EVT TOKEN ");
    for(int i=0;i<int'(vlen[id]);i++) $write("%02x",vb[int'(vst[id])+i]);
    $write("\n");
    $fflush();
  endtask
  task automatic forward(input int token,input int pos);
    real u,vv,theta,cs,sn,smax,den,z;
    int b,kv;
    for(int i=0;i<H;i++) x[i]=emb[token*H+i];
    for(int layer=0;layer<L;layer++) begin
      b=1+layer*9;
      rms(b);
      load(b+1,H*H);matvec(H,H,0);
      load(b+2,192*H);matvec(192,H,1);
      load(b+3,192*H);matvec(192,H,2);
      for(int head=0;head<9;head++)
        for(int j=0;j<32;j++) begin
          theta=pos*inv[j];cs=$cos(theta);sn=$sin(theta);
          u=q[head*64+j];vv=q[head*64+j+32];
          q[head*64+j]=u*cs-vv*sn;q[head*64+j+32]=vv*cs+u*sn;
        end
      for(int head=0;head<3;head++)
        for(int j=0;j<32;j++) begin
          theta=pos*inv[j];cs=$cos(theta);sn=$sin(theta);
          u=k[head*64+j];vv=k[head*64+j+32];
          k[head*64+j]=u*cs-vv*sn;k[head*64+j+32]=vv*cs+u*sn;
        end
      for(int i=0;i<192;i++) begin kc[layer][pos][i]=k[i];vc[layer][pos][i]=val[i];end
      for(int head=0;head<9;head++) begin
        kv=(head/3)*64;smax=-1.0e100;
        for(int t=0;t<=pos;t++) begin
          z=0.0;
          for(int j=0;j<64;j++) z+=q[head*64+j]*kc[layer][t][kv+j];
          score[t]=z/8.0;if(score[t]>smax) smax=score[t];
        end
        den=0.0;
        for(int t=0;t<=pos;t++) begin score[t]=$exp(score[t]-smax);den+=score[t];end
        for(int j=0;j<64;j++) begin
          z=0.0;
          for(int t=0;t<=pos;t++) z+=(score[t]/den)*vc[layer][t][kv+j];
          att[head*64+j]=z;
        end
      end
      for(int i=0;i<H;i++) a[i]=att[i];
      load(b+4,H*H);matvec(H,H,6);
      rms(b+5);
      load(b+6,F*H);matvec(F,H,4);
      load(b+7,F*H);matvec(F,H,5);
      for(int i=0;i<F;i++) up[i]*=gate[i]/(1.0+$exp(-gate[i]));
      load(b+8,H*F);
      for(int i=0;i<H;i++) begin
        z=0.0;for(int j=0;j<F;j++) z+=w[i*F+j]*up[j];
        x[i]+=z;
      end
    end
    rms(271);
    best=-1.0e100;next_id=0;
    for(int id=0;id<V;id++) begin
      z=0.0;for(int j=0;j<H;j++) z+=emb[id*H+j]*a[j];
      if(z>best) begin best=z;next_id=id;end
    end
  endtask
  initial begin
    if(!$value$plusargs("prompt=%s",prompt)) $fatal(1,"use +prompt=TEXT");
    if(!$value$plusargs("max_new=%d",max_new)) max_new=16;
    if(!$value$plusargs("model=%s",model_path)) model_path="data/model.safetensors";
    if(max_new<1||max_new>S) $fatal(1,"invalid max_new");
    machine_mode=$test$plusargs("machine");
    $readmemh("data/tensor_offset.mem",off);
    $readmemh("data/byte_id.mem",byte_id);
    $readmemh("data/merge_left.mem",ml);$readmemh("data/merge_right.mem",mr);
    $readmemh("data/merge_result.mem",mo);$readmemh("data/merge_rank.mem",mk);
    $readmemh("data/vocab_start.mem",vst);$readmemh("data/vocab_len.mem",vlen);
    $readmemh("data/vocab_bytes.mem",vb);
    $readmemh("data/unicode_start.mem",ustart);
    $readmemh("data/unicode_end.mem",uend);
    $readmemh("data/unicode_kind.mem",ukind);
    pow2[127]=1.0;
    for(int i=128;i<256;i++) pow2[i]=pow2[i-1]*2.0;
    for(int i=126;i>=0;i--) pow2[i]=pow2[i+1]/2.0;
    for(int i=0;i<32;i++) inv[i]=1.0/(10000.0**(2.0*i/64));
    tokenize();
    if($test$plusargs("tokenize_only")) begin
      $write("TOKENS:");
      for(int ti=0;ti<ntok;ti++) $write(" %0d",toks[ti]);
      $write("\n");
    end else begin
    if(machine_mode) begin
      $display("EVT PROMPT %0d",ntok);
      $display("EVT LOADING");
      $fflush();
    end
    fd=$fopen(model_path,"rb");if(fd==0) $fatal(1,"cannot open %s",model_path);
    for(int row=0;row<V;row+=F) begin
      read_at(off[0]+longint'(row)*H*4,F*H);
      for(int i=0;i<F*H;i++) emb[row*H+i]=f32(word_at(i*4));
    end
    if(ntok+max_new>S) $fatal(1,"prompt plus output exceeds %0d tokens",S);
    if(!machine_mode) $display("PROMPT TOKENS: %0d",ntok);
    for(int pos=0;pos<ntok;pos++) begin
      if(!machine_mode) $display("PROGRESS: prompt token %0d/%0d",pos+1,ntok);
      else $display("EVT PROCESSING %0d %0d",pos+1,ntok);
      $fflush();
      forward(toks[pos],pos);
      if(machine_mode) begin
        $display("EVT PREFILL %0d %0d",pos+1,ntok);
        $fflush();
      end
    end
    if(!machine_mode) begin $write("OUTPUT: ");$fflush();end
    for(int gen=0;gen<max_new;gen++) begin
      if(next_id==0) break;
      if(machine_mode) emit_machine(next_id);
      else emit(next_id);
      if(gen+1<max_new) forward(next_id,ntok+gen);
    end
    if(machine_mode) begin $display("EVT DONE");$fflush();end
    else $write("\n");
    $fclose(fd);
    end
    $finish;
  end
endmodule
