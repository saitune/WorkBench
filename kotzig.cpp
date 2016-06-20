#include<stdio.h>
#include<algorithm>
#include<cstdint>
#include<iostream>
#include<bitset>
#include<time.h>
#include<vector>



#if 0
template<int N>int factorial(void)
{
  return (N==1)? 1:N*factorial<N-1>();
}
#endif



#if 0
const int N=5;
const int NumOfHamilton=1*2*3*4/2;
#endif

#if 0
const int N=7;
const int NumOfHamilton=1*2*3*4*5*6/2;
const int HdLimit=0;
#endif

#if 1
const int N = 9;
const int NumOfHamilton = 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 / 2;
// hd上限, 無し=0
const int HdLimit=0;
#endif



const int DudeneySize = (N - 1) * (N - 2) / 2;
//const int NumOfHamilton=factorial<N-1>()/2;
const int NumOfTwopath = N * DudeneySize;
const int SizeOfTwopath = NumOfTwopath / 64 + 1;
const int NumOfOnepath = (N * N - N) / 2;
const int SizeOfOnepath = NumOfOnepath / 64 + 1;
const int SizeOfDecomposition = N / 2;
const int NumOfDecomposition = DudeneySize / SizeOfDecomposition;

const int ExtractLevel=7;

clock_t start_time;
uint64_t Count[NumOfDecomposition],RecCall=0;



void print(const int array[][N],const int size)
{
  for(int i=0;i<size;i++){
    for(int j=0;j<N;j++) printf("%d",array[i][j]);
    printf("\n");
  }
}



// ハミルトンサイクル生成
void make_hamilton(const int lv,int& hn,int h[N],bool vflg[N],int hs[][N])
{
  if(lv==N){
    if(h[1]<h[N-1]){
      for(int j=0;j<N;j++) hs[hn][j]=h[j];
      hn++;
    }
  }
  else {
    for(int i=1;i<N;i++) if(vflg[i]==false){
        vflg[i]=true;
        h[lv]=i;
        make_hamilton(lv+1,hn,h,vflg,hs);
        vflg[i]=false;
        h[lv]=0;
      }
  }
}



// ビットフィールド生成
void make_twopath_bf(const int hs[][N],uint64_t Twopath_bf[][SizeOfTwopath],
                     const int f[N][N][N])
{
  for(int i=0;i<NumOfHamilton;i++) for(int j=0;j<N;j++){
      const int l=hs[i][j];
      const int m=hs[i][(j+1)%N];
      const int n=hs[i][(j+2)%N];
      //確答ビットを埋める
      Twopath_bf[i][f[l][m][n]/64]|=(1ULL<<f[l][m][n]%64);
    }
}



void make_onepath_bf(const int hs[][N],uint64_t Onepath_bf[][SizeOfOnepath],
                     const int g[][N])
{
  for(int i=0;i<NumOfHamilton;i++) for(int j=0;j<N;j++){
      const int l=hs[i][j];
      const int m=hs[i][(j+1)%N];
      Onepath_bf[i][g[l][m]/64]|=(1ULL<<g[l][m]%64);
    }
}



// 末尾に続く0の数を返す
int number_of_trailing_zeros(uint64_t bf)
{
  if(bf==0) return 64;
  int n=1;
  if((bf & 0x00000000ffffffff) == 0){n += 32;bf >>= 32;}
  if((bf & 0x000000000000ffff) == 0){n += 16;bf >>= 16;}
  if((bf & 0x00000000000000ff) == 0){n += 8;bf >>= 8;}
  if((bf & 0x000000000000000f) == 0){n += 4;bf >>= 4;}
  if((bf & 0x0000000000000003) == 0){n += 2;bf >>= 2;}
  return n-static_cast<int>(bf & 1);
}



#if 0
// 末尾に続く1の数を返す
int get_minimum_zero_bit(uint64_t bf)
{
  int n;
  if(bf == 0)return 0;
  n = 1;
  if((bf & 0x00000000ffffffff) == 0x00000000ffffffff){n += 32;bf >>= 32;}
  if((bf & 0x000000000000ffff) == 0x000000000000ffff){n += 16;bf >>= 16;}
  if((bf & 0x00000000000000ff) == 0x00000000000000ff){n += 8;bf >>= 8;}
  if((bf & 0x000000000000000f) == 0x000000000000000f){n += 4;bf >>= 4;}
  if((bf & 0x0000000000000003) == 0x0000000000000003){n += 2;bf >>= 2;}
  if((bf & 1) == 0){n--;}
  return n ;
}
#endif



int nto(const int size,const uint64_t bf[])
{
  for(int i=0;i<size;i++){
    const int n=number_of_trailing_zeros(~bf[i]);
    if(n<64){
      return 64*i+n;
      break;
    }
  }
  return -1;
}



int ntz(const int size,const uint64_t bf[])
{
  for(int i=0;i<size;i++){
    const int n=number_of_trailing_zeros(bf[i]);
    if(n<64){
      return 64*i+n;
      break;
    }
  }
  return -1;
}





// hs,bfを最少ビットごとに分類
void classify_by_bit
(const int hs[][N],const uint64_t Twopath_bf[][SizeOfTwopath],
 const uint64_t Onepath_bf[][SizeOfOnepath],int class_size[],
 int classified_hs[NumOfTwopath][NumOfHamilton][N],
 uint64_t classified_Twopath_bf[][NumOfHamilton][SizeOfTwopath],
 uint64_t classified_Onepath_bf[][NumOfHamilton][SizeOfOnepath])
{
  for(int i=0;i<NumOfHamilton;i++){//ハミルトンサイクルすべてを探索
    const int min_bit=ntz(SizeOfOnepath,Onepath_bf[i]);
    //最小のビットを返す = 末尾に続く0の数を返す
    //onepath の最小ビットに対応する配列要素にhs,onepath,twopathの情報を格納
    for(int j=0;j<N;j++)
      classified_hs[min_bit][class_size[min_bit]][j]=hs[i][j];
    for(int j=0;j<SizeOfTwopath;j++)
      classified_Twopath_bf[min_bit][class_size[min_bit]][j]=Twopath_bf[i][j];
    for(int j=0;j<SizeOfOnepath;j++)
      classified_Onepath_bf[min_bit][class_size[min_bit]][j]=Onepath_bf[i][j];
    class_size[min_bit]++;
  }
}



//ビット照合(互いに素かどうか)
bool test_twopath_bit(const uint64_t bf_a[SizeOfTwopath],
                      const uint64_t bf_b[SizeOfTwopath])
{
  for(int i=0;i<SizeOfTwopath;i++)if((bf_a[i]&bf_b[i])!=0)return false;
  return true;
}



bool test_onepath_bit(const uint64_t bf_a[SizeOfOnepath],
                      const uint64_t bf_b[SizeOfOnepath])
{
  for(int i=0;i<SizeOfOnepath;i++)if((bf_a[i]&bf_b[i])!=0)return false;
  return true;
}



//ビットセット
void set_twopath_bit(uint64_t check_Twopath_bf[SizeOfTwopath],
                     const uint64_t Twopath_bf[SizeOfTwopath])
{
  for (int i = 0; i < SizeOfTwopath; i++)check_Twopath_bf[i] |= Twopath_bf[i];
}



void set_onepath_bit(uint64_t check_Onepath_bf[SizeOfOnepath],
                     const uint64_t Onepath_bf[SizeOfOnepath])
{
  for(int i = 0;i < SizeOfOnepath; i++)check_Onepath_bf[i] |= Onepath_bf[i];
}



//ビットリセット
void reset_twopath_bit(uint64_t check_Twopath_bf[SizeOfTwopath],
                       const uint64_t Twopath_bf[SizeOfTwopath])
{
  for (int i = 0; i < SizeOfTwopath; i++)check_Twopath_bf[i] &= ~Twopath_bf[i];
}



void reset_onepath_bit(uint64_t check_Onepath_bf[SizeOfOnepath],
                       const uint64_t Onepath_bf[SizeOfOnepath])
{
  for (int i = 0; i < SizeOfOnepath; i++)check_Onepath_bf[i] &= ~Onepath_bf[i];
}



//ハミルトンサイクルの標準化
void normalize(const int input[N],int output[N])
{
  int i;
  for (i = 0; input[i] != 0; i++);
  for (int j = 0; j < N; j++)output[j] = input[(i + j) % N];
  if (output[1] > output[N - 1])std::reverse(output + 1,output + N);
}



//比較関数
int cmp(const int h[],const int g[])
{
  for (int i = 0; i < N; i++){
    if (h[i] < g[i])return -1;
    else if (h[i] > g[i])return 1;
  }
  return 0;
}



// 同形判定
bool is_normal(const int num_of_hd,const int dudeney[][N],int trns[][N],
               int& trns_size)
{
#if 0
  //メモ実装前

  //ベクター生成
  const int hn = (num_of_hd + 1) * SizeOfDecomposition;
  std::vector<std::vector<std::vector<int> > >
    f(num_of_hd + 1,std::vector<std::vector<int>>(SizeOfDecomposition,
                                                  std::vector<int>(N,0)));
  //
  int sigma[N];
  int temp[DudeneySize][N];
  std::fill_n(temp[0],DudeneySize * N,0);
  for (int i = 0; i < hn; i++){
    for (int d = 0; d < N; d++){
      //置換を生成
      for (int j = 0; j < N; j++)sigma[dudeney[i][j]]=(j + d)%N;

      //tempに置換を適用後のハミルトンサイクルを格納
      for (int j = 0; j < hn; j++)
        for (int k = 0; k < N; k++)temp[j][k]=sigma[dudeney[j][k]];

      //tempに格納されているハミルトンサイクルを標準化してfに代入
      for (int j = 0; j < hn; j++)
        normalize(temp[j],
                  &f[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);

      //同一HamiltonDecomposition中のハミルトンサイクルをソート
      for (int j = 0; j <= num_of_hd; j++)std::sort(f[j].begin(),f[j].end());

      //HamiltonDecompositionのソート
      std::sort(f.begin(),f.end());

      //入力された集合が標準形(最小)でなければfalse
      //最小であれば次へ
      for (int j = 0; j < hn; j++){
        int k= cmp(dudeney[j],&f[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);
        if (k == 1)return false;
        else if (k == -1)goto next1;
      }
    next1:;
    }
    for (int d = 0; d < N; d++){
      //置換を生成
      for (int j = 0; j < N; j++)sigma[dudeney[i][j]]=(d - j + N)%N;

      //tempに置換を適用後のハミルトンサイクルを格納
      for (int j = 0; j < hn; j++)
        for (int k = 0; k < N; k++)temp[j][k]=sigma[dudeney[j][k]];

      //tempに格納されているハミルトンサイクルを標準化してfに代入
      for (int j = 0; j < hn; j++)
        normalize(temp[j],&f[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);

      //同一HamiltonDecomposition中のハミルトンサイクルをソート
      for (int j = 0; j <= num_of_hd; j++)std::sort(f[j].begin(),f[j].end());

      //HamiltonDecompositionのソート
      std::sort(f.begin(),f.end());

      //入力された集合が標準形(最小)でなければfalse
      //最小であれば次へ
      for (int j = 0; j < hn; j++){
        int k = cmp(dudeney[j],&f[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);
        if (k == 1)return false;
        else if (k == -1)goto next2;
      }
    next2:;
    }
  }

#else
  // メモ使用版.
  
  // 初期num_of_hdが0の時,hn採用
  const int hn = (num_of_hd + 1) * SizeOfDecomposition;

  // ベクター生成
  std::vector<std::vector<std::vector<int> > >
    h(num_of_hd+1,std::vector<std::vector<int>>(SizeOfDecomposition,std::vector<int>(N,0)));

  int temp_for_h[DudeneySize][N];

  std::fill_n(temp_for_h[0],DudeneySize * N,0);

  // trnsに保存されている置換を試す
  for (int i = 0; i < trns_size; i++){
    
    //射影
    for (int j = 0; j < hn; j++)
      for (int k = 0; k < N; k++)temp_for_h[j][k] = trns[i][dudeney[j][k]];
    
    //標準化
    for (int j = 0; j < hn; j++)
      for (int k = 0; k < N; k++)
        normalize(temp_for_h[j],
                  &h[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);
    
    //hd内をソート
    for (int j = 0; j <= num_of_hd; j++)std::sort(h[j].begin(),h[j].end());
    
    //全体をソート
    std::sort(h.begin(),h.end());

    //オリジナルと変換後を比較
    for (int j = 0; j < hn; j++){
      const int k = cmp(dudeney[j],&h[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);
      
      //オリジナルが、大きい(k = 1),小さい(k = -1)
      if (k == 1)return false;
      else if (k == -1)break;
    }
  }

  //新しく追加されたhdについて、置換を求める
  std::vector<std::vector<std::vector<int>>> f(num_of_hd+1,std::vector<std::vector<int>>(SizeOfDecomposition,std::vector<int>(N,0)));
  int sigma[N];
  int temp[DudeneySize][N];
  int hd_temp[SizeOfDecomposition][N];
  std::fill_n(hd_temp[0],SizeOfDecomposition * N,0);
  std::fill_n(temp[0],DudeneySize * N,0);

  for (int i = 0; i < SizeOfDecomposition; i++){
    for (int d = 0; d < N; d++){

      //置換を生成
      for (int j = 0; j < N; j++)sigma[dudeney[num_of_hd * SizeOfDecomposition + i][j]] = (j + d) % N;
      
      //射影
      for (int j = 0; j < hn; j++)for (int k = 0; k < N; k++)temp[j][k] = sigma[dudeney[j][k]];
      
      //標準化
      for (int j = 0; j < hn; j++)normalize(temp[j],&f[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);
      
      //hd内をソート
      for (int j = 0; j <= num_of_hd; j++){
        std::sort(f[j].begin(),f[j].end());

        //ソート済みの新hdをhd_tempに代入
        if (j == num_of_hd)for (int k = 0; k < SizeOfDecomposition; k++)for (int l = 0; l < N; l++)hd_temp[k][l] = f[num_of_hd][k][l];
      
      }

      //全体をソート
      std::sort(f.begin(),f.end());

      //新hd'とhd(0)が等しければメモ
      for (int j = 0; j < SizeOfDecomposition; j++){
        const int k = cmp(hd_temp[j],dudeney[j]);
        if (k == 1 || k == -1) break;

        //等しかったらtrnsを更新
        if (j == SizeOfDecomposition-1){
          for (int l = 0; l < N; l++)trns[trns_size][l] = sigma[l];
          trns_size++;
        }
      }

      //fとオリジナルを比較
      for (int j = 0; j < hn; j++){
        const int k = cmp(dudeney[j],&f[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);
        if (k == 1)return false;
        else if (k == -1)break;
      }


    }
    for (int d = 0; d < N; d++){

      //置換を生成
      for (int j = 0; j < N; j++)sigma[dudeney[num_of_hd * SizeOfDecomposition + i][j]] = (d - j + N) % N;
      
      //射影
      for (int j = 0; j < hn; j++)for (int k = 0; k < N; k++)temp[j][k] = sigma[dudeney[j][k]];
      
      //標準化
      for (int j = 0; j < hn; j++)normalize(temp[j],&f[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);
      
      //hd内をソート
      for (int j = 0; j <= num_of_hd; j++){
        std::sort(f[j].begin(),f[j].end());

        //ソート済み新hdをhd_tempに代入
        if (j == num_of_hd)for (int k = 0; k < SizeOfDecomposition; k++)for (int l = 0; l < N; l++)hd_temp[k][l] = f[num_of_hd][k][l];
      }

      //全体をソート
      std::sort(f.begin(),f.end());

      //hd_tempとhd(0)が等しければメモ
      for (int j = 0; j < SizeOfDecomposition; j++){
        const int k = cmp(hd_temp[j],dudeney[j]);
        if (k == 1 || k == -1) break;

        //等しかったらtrnsを更新
        if (j == SizeOfDecomposition-1){
          for (int l = 0; l < N; l++)trns[trns_size][l] = sigma[l];
          trns_size++;
        }
      }

      //fとオリジナルを比較
      for (int j = 0; j < hn; j++){
        const int k = cmp(dudeney[j],&f[j / SizeOfDecomposition][j % SizeOfDecomposition][0]);
        if (k == 1)return false;
        else if (k == -1)break;
      }
    }
  }
#endif
  return true;
}



/* 入力 分類済みのhs,onepath_bf,twopath_bf
   check_twopathに含まれないビットのみで構成されるハミルトンサイクルと
   それに対応するビットフィールドをnewClassfied_hs,
   newClassified_Onepath_bf, newClassified_Twopath_bfに格納 */
void extract
(const int class_size[],const int classified_hs[][NumOfHamilton][N],
 const uint64_t classified_Onepath_bf[][NumOfHamilton][SizeOfOnepath],
 const uint64_t classified_Twopath_bf[][NumOfHamilton][SizeOfTwopath],
 const uint64_t check_twopath_bf[],int newClassified_hs[][NumOfHamilton][N],
 uint64_t newClassified_Onepath_bf[][NumOfHamilton][SizeOfOnepath],
 uint64_t newClassified_Twopath_bf[][NumOfHamilton][SizeOfTwopath],
 int newClass_size[])
{
  /* newClassified_hs, newClassified_Onepath_bf,
     newClassified_Twopath_bfは初期化不要 */
     
  std::fill_n(newClass_size,NumOfOnepath,0);
    
  for(int i=0; i<NumOfOnepath; i++)
    for(int j=0; j<class_size[i]; j++)
      if(test_twopath_bit(check_twopath_bf,classified_Twopath_bf[i][j])){
        std::copy(classified_hs[i][j],classified_hs[i][j]+N,
                  newClassified_hs[i][newClass_size[i]]);
        std::copy(classified_Onepath_bf[i][j],
                  classified_Onepath_bf[i][j]+SizeOfOnepath,
                  newClassified_Onepath_bf[i][newClass_size[i]]);
        std::copy(classified_Twopath_bf[i][j],
                  classified_Twopath_bf[i][j]+SizeOfTwopath,
                  newClassified_Twopath_bf[i][newClass_size[i]]);
        newClass_size[i]++;
      }
}



// デュードニー集合生成
void make_dudeney
(const int hn,int dudeney[DudeneySize][N],
 uint64_t check_Twopath_bf[SizeOfTwopath],
 uint64_t check_Onepath_bf[SizeOfOnepath],int class_size[],
 int classified_hs[][NumOfHamilton][N],
 uint64_t classified_Twopath_bf[][NumOfHamilton][SizeOfTwopath],
 uint64_t classified_Onepath_bf[][NumOfHamilton][SizeOfOnepath],int trns[][N],
 int trns_size)
{
  // テスト
  if(RecCall%1000000==0){
    fprintf(stderr,"RecCall=%ld\n",RecCall);
    for(int i=0; i<NumOfDecomposition; i++)
      fprintf(stderr,"HD %d => %ld\n",i+1,Count[i]);
    fprintf(stderr,"\n");
  }
  // if(RecCall==10000000) exit(0);

#if 0
  // テスト
  if(Count[0]==2){
    for(int i=0; i<NumOfDecomposition; i++)
      fprintf(stderr,"HD %d => %d\n",i+1,Count[i]);
    exit(0);
  }
#endif
  
  if(HdLimit && hn/SizeOfDecomposition==HdLimit) return;
  ++RecCall;

  const int min_bit = nto(SizeOfOnepath,check_Onepath_bf);
  // onepathにおいて最小の0であるビットを求める = 最小から連続する1の数を返す

#if 0 //デバッグ用表示
  printf("dudeney\n");
  print(dudeney);
  printf("\n");
  printf("check_onepath_bf\n");
  for(int i=SizeOfOnepath-1;0<=i;i--)
    std::cout << static_cast<std::bitset<64> >(check_Onepath_bf[i]);
  printf("\n");
  printf("check_twopath_bf\n");
  for(int i=SizeOfTwopath-1;0<=i;i--)
    std::cout << static_cast<std::bitset<64> >(check_Twopath_bf[i]);
  printf("\n");
  printf("min_bit = %d\n",min_bit);
  getchar();
  getchar();
#endif

  for (int i = 0; i < class_size[min_bit]; i++){
    /* decompositionの最初のサイクルが1つ前のdecompositionの最初のサイ
       クルより小さかったら枝刈り */
    if(hn != 0 && hn % SizeOfDecomposition == 0
       && cmp(dudeney[hn-SizeOfDecomposition],classified_hs[min_bit][i])==1)
      continue;

    // 新たなサイクルのonepath,twopathが互いに素であるとき
    if (test_onepath_bit(check_Onepath_bf,classified_Onepath_bf[min_bit][i]) 
        && test_twopath_bit(check_Twopath_bf,classified_Twopath_bf[min_bit][i])
        ){
      // onepath,twopathビットフィールドをセット
      set_onepath_bit(check_Onepath_bf,classified_Onepath_bf[min_bit][i]);
      set_twopath_bit(check_Twopath_bf,classified_Twopath_bf[min_bit][i]);

      // dudeneyに追加
      for (int j = 0; j < N; j++)dudeney[hn][j] = classified_hs[min_bit][i][j];

      // decompositionの最終サイクルのとき
      if (hn % SizeOfDecomposition == SizeOfDecomposition - 1){

        int new_trns_size = trns_size;

        // 同形判定
        if(!is_normal(hn / SizeOfDecomposition,dudeney,trns,new_trns_size))
          goto next;

        // onepath リセット
        uint64_t NewCheck_Onepath_bf[SizeOfOnepath];
        std::fill_n(NewCheck_Onepath_bf,SizeOfOnepath,0);
        Count[hn / SizeOfDecomposition]++;

        if(hn<=ExtractLevel){
          // extractする場合
          int newClass_size[NumOfOnepath],
            newClassified_hs[NumOfOnepath][NumOfHamilton][N];
          uint64_t newClassified_Onepath_bf[NumOfOnepath][NumOfHamilton]
            [SizeOfOnepath],
            newClassified_Twopath_bf[NumOfOnepath][NumOfHamilton]
            [SizeOfTwopath];

          // printf("extract\n");
          extract(class_size,classified_hs,classified_Onepath_bf,
                  classified_Twopath_bf,check_Twopath_bf,
                  newClassified_hs,newClassified_Onepath_bf,
                  newClassified_Twopath_bf,newClass_size);
          
          // printf("finish extract\n");

          make_dudeney(hn + 1,dudeney,check_Twopath_bf,NewCheck_Onepath_bf,
                       newClass_size,newClassified_hs,newClassified_Twopath_bf,
                       newClassified_Onepath_bf,trns,new_trns_size);
        }

        // extractしない場合
        else make_dudeney(hn + 1,dudeney,check_Twopath_bf,NewCheck_Onepath_bf,
                          class_size,classified_hs,classified_Twopath_bf,
                          classified_Onepath_bf,trns,new_trns_size);
      }

      // decompositionの最終サイクルではないとき
      else{
        // printf("else\n");

        if(hn<=ExtractLevel){
          // extractする場合
          int newClass_size[NumOfOnepath],
            newClassified_hs[NumOfOnepath][NumOfHamilton][N];
          uint64_t newClassified_Onepath_bf[NumOfOnepath][NumOfHamilton]
            [SizeOfOnepath],
            newClassified_Twopath_bf[NumOfOnepath][NumOfHamilton]
            [SizeOfTwopath];
          
          // printf("extract2\n");
          extract(class_size,classified_hs,classified_Onepath_bf,
                  classified_Twopath_bf,check_Twopath_bf,newClassified_hs,
                  newClassified_Onepath_bf,newClassified_Twopath_bf,
                  newClass_size);

          // printf("finish extract2\n");
          make_dudeney(hn + 1,dudeney,check_Twopath_bf,check_Onepath_bf,
                       newClass_size,newClassified_hs,newClassified_Twopath_bf,
                       newClassified_Onepath_bf,trns,trns_size);
        }

        // extractしない場合.
        else make_dudeney(hn + 1,dudeney,check_Twopath_bf,check_Onepath_bf,
                          class_size,classified_hs,classified_Twopath_bf,
                          classified_Onepath_bf,trns,trns_size);
      }
    next:;
      // ビットフィールド、dudeneyをリセット
      reset_onepath_bit(check_Onepath_bf,classified_Onepath_bf[min_bit][i]);
      reset_twopath_bit(check_Twopath_bf,classified_Twopath_bf[min_bit][i]);
      for (int j = 0; j < N; j++)dudeney[hn][j] = 0;
    }
  }
}



void show_bit(uint64_t sorted_bf[][NumOfHamilton][SizeOfTwopath])
{
  int b=0;
  for(int i=0;i<NumOfTwopath;i++){
    for(int j=0;j<NumOfHamilton;j++){
      if((sorted_bf[i][j][i/64]&(1ULL<<i%64))!=0){
        //std::cout << static_cast<std::bitset<64> >(sorted_bf[i][j][0]) << std::endl;
        b++;
      }
    }
  }
  printf("b=%d\n",b);
}



int main(void)
{
  start_time=clock();
  std::fill_n(Count,NumOfDecomposition,0);
  //set f
  int f[N][N][N];
  int l = 0;
  for (int i = 0; i < N - 1; i++)for (int k = i + 1; k < N; k++)for (int j = 0; j < N; j++)if (j != i && j != k){
          //twopath 表示
          //printf("%d-%d-%d => %d\n",i,j,k,l);
          f[i][j][k]=f[k][j][i] = l++;
        }

#if 0
  printf("f set\n");
#endif

  //set g
  int g[N][N];
  l = 0;
  for (int i = 0; i < N - 1; i++)for (int j = i + 1; j < N; j++){
      //onepath 表示
      //printf("%d-%d => %d\n",i,j,l);
      g[i][j]=g[j][i] = l++;
    }

#if 0
  printf("g set\n");
#endif

  // set hs
  int hs[NumOfHamilton][N];
  std::fill_n(hs[0],NumOfHamilton*N,0);
#if 0
  printf("hs set\n");
#endif

  // set vflg
  bool vflg[N];
  std::fill_n(vflg,N,false);
#if 0
  printf("vflg set\n");
#endif

  // make hs
  int h[N];
  std::fill_n(h,N,0);
  int hn=0;
  make_hamilton(1,hn,h,vflg,hs);
#if 0
  printf("hs made\n");
#endif

  // set Twopath bf
  uint64_t Twopath_bf[NumOfHamilton][SizeOfTwopath];
  std::fill_n(Twopath_bf[0],NumOfHamilton * SizeOfTwopath,0);
  make_twopath_bf(hs,Twopath_bf,f);
#if 0
  printf("Twopath_bf set\n");
#endif

  // set Onepath bf
  uint64_t Onepath_bf[NumOfHamilton][SizeOfOnepath];
  std::fill_n(Onepath_bf[0],NumOfHamilton * SizeOfOnepath,0);
  make_onepath_bf(hs,Onepath_bf,g);
#if 0
  printf("Onepath_bf set\n");
#endif

  //set dudeney
  int dudeney[DudeneySize][N];
  std::fill_n(dudeney[0],DudeneySize*N,0);
#if 0
  printf("dudeney set\n");
#endif

  //classified
  int classified_hs[NumOfOnepath][NumOfHamilton][N];
  uint64_t classified_Twopath_bf[NumOfOnepath][NumOfHamilton][SizeOfTwopath];
  uint64_t classified_Onepath_bf[NumOfOnepath][NumOfHamilton][SizeOfOnepath];
  std::fill_n(classified_hs[0][0],NumOfOnepath*NumOfHamilton*N,0);
  std::fill_n(classified_Twopath_bf[0][0],NumOfOnepath * NumOfHamilton * SizeOfTwopath,0);
  std::fill_n(classified_Onepath_bf[0][0],NumOfOnepath * NumOfHamilton * SizeOfOnepath,0);
  int class_size[NumOfOnepath];
  std::fill_n(class_size,NumOfOnepath,0);
#if 0
  printf("classified set\n");
#endif

  classify_by_bit(hs,Twopath_bf,Onepath_bf,class_size,classified_hs,
                  classified_Twopath_bf,classified_Onepath_bf);
#if 0
  printf("classified_by_bit finished\n");
#endif


  //set check_bf
  uint64_t check_Twopath_bf[SizeOfTwopath];
  std::fill_n(check_Twopath_bf,SizeOfTwopath,0);
  uint64_t check_Onepath_bf[SizeOfOnepath];
  std::fill_n(check_Onepath_bf,SizeOfOnepath,0);
#if 0
  printf("check_bf set\n");
#endif


  //test
  /*
    for(int i=0;i<NumOfOnepath;i++)for(int j=0;j<class_size[i];j++)for(int k=0;k<SizeOfOnepath;k++){
    std::cout << static_cast<std::bitset<64> >(classified_Onepath_bf[i][j][k]) << std::endl;
    }
  */

  //テスト表示 hs classified_Onepath_bf classified_Twopath_bf
#if 0
  for(int i=0;i<NumOfOnepath;i++)for(int j=0;j<class_size[i];j++){
      printf("hs\n");
      for(int k=0;k<N;k++)printf("%d",classified_hs[i][j][k]);
      printf("\n");
      printf("onepath_bf\n");
      for(int k=0;k<SizeOfOnepath;k++)std::cout << static_cast<std::bitset<64> >(classified_Onepath_bf[i][j][k]);
      printf("\n");
      printf("twopath_bf\n");
      for(int k=SizeOfTwopath-1;0<=k;k--)std::cout << static_cast<std::bitset<64> >(classified_Twopath_bf[i][j][k]);
      printf("\n");
      getchar();
      getchar();
    }
#endif

  //ハミルトンサイクルの一つ目を01234....n-1に固定
  for(int i=0;i<N;i++)dudeney[0][i]=i;

  //classified_hs からサイクル 01234....n-1を探索し、それに対応するビットをcheck_bfにセット
  //01234...n-1はclassified_hs[0][0]にあることは確定
  set_onepath_bit(check_Onepath_bf,classified_Onepath_bf[0][0]);
  set_twopath_bit(check_Twopath_bf,classified_Twopath_bf[0][0]);

  int newClass_size[NumOfOnepath],
    newClassified_hs[NumOfOnepath][NumOfHamilton][N];
  uint64_t
    newClassified_Onepath_bf[NumOfOnepath][NumOfHamilton][SizeOfOnepath],
    newClassified_Twopath_bf[NumOfOnepath][NumOfHamilton][SizeOfTwopath];

  //extractをかける
  extract(class_size,classified_hs,classified_Onepath_bf,classified_Twopath_bf,
          check_Twopath_bf,newClassified_hs,newClassified_Onepath_bf,
          newClassified_Twopath_bf,newClass_size);

  //trns trns_size 初期化

  int trns[DudeneySize * 2 * N][N];
  std::fill_n(trns[0],DudeneySize * 2 * N * N,0);

  make_dudeney(1,dudeney,check_Twopath_bf,check_Onepath_bf,newClass_size,
               newClassified_hs,newClassified_Twopath_bf,
               newClassified_Onepath_bf,trns,0);
#if 0
  printf("makeDudeney finished\n");
#endif
  for (int i = 0; i < NumOfDecomposition; i++)
    printf("HD %d => %ld\n",i + 1,Count[i]);
  printf("RecCall=%ld\n",RecCall);
  printf("%f sec\n",static_cast<double>(clock()-start_time)/CLOCKS_PER_SEC);
}
