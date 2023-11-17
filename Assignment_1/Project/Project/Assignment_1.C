#include <stdio.h>
#include <stdlib.h>

//// 파라미터 ////
#define input_size 16*16
#define output_size 7
#define hidden_size 1000
int learning_cycle = 100;
int data_set=10;
int test_set = 20;
float learning_rate = 0.05;

///// 신경망 구현 ////////
float input_layer[input_size]={0,};
float output_layer[output_size]={0,};
float hidden_layer1[hidden_size]={0,};
float hidden_layer2[hidden_size]={0,};

float w1[hidden_size][input_size]={0,};
float w2[hidden_size][hidden_size]={0,};
float w3[output_size][hidden_size]={0,};

float hidden_output1[hidden_size]={0,};
float hidden_output2[hidden_size]={0,};
float output_output[output_size]={0,};

float output_error[output_size]={0,};
float hidden_error1[hidden_size]={0,};
float hidden_error2[hidden_size]={0,};

float img[input_size]={0,};

//// 가중치 초기화 ////
void wreset()
{
    int i, j;

    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<input_size;j++)
        {
            w1[i][j]=rand()/RAND_MAX;
        }
    }

    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            w2[i][j]=rand()/RAND_MAX;
        }
    }

    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            w3[i][j]=rand()/RAND_MAX;
        }
    }
}

//2차원 배열 크기 알아내는법
//int row_A = sizeof(A) / sizeof(A[0]);
//int col_A = sizeof(A[0])/sizeof(float);

// 파이썬에서 배열을 CSV 형태로 저장한 뒤 불러와서 사용

float relu(float x)
{
    float o = max(0,x);
    return o;
}

void FP()
{
    int i,j;

    //// input_layer////
    for(i=0;i<input_size;i++)
    {
        input_layer[i]=img[i];
    }

    //// hidden_layer1 ////
    //// w1과 input_layer를 행렬곱해 hidden_layer1에 저장해준다
    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<input_size;j++)
        {
            hidden_layer1[i] += (w1[i][j]*input_layer[j]);
        }
    }
    //// ReLU연산을 한 뒤 hidden_output1에 저장해준다
    for(i=0;i<hidden_size;i++)
    {
        hidden_output1[i] = relu(hidden_layer1[i]);
    }

    //// hidden_layer2 ////
    //// w2와 hidden_output1을 행렬곱해 hidden_layer2에 저장해준다
    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            hidden_layer2[i] += (w2[i][j]*hidden_output1[j]);
        }
    }
    //// ReLU연산을 한 뒤 hidden_output2에 저장해준다
    for(i=0;i<hidden_size;i++)
    {
        hidden_output2[i] = relu(hidden_layer2[i]);
    }

    //// output_layer ////
    //// w3과 hidden_output2를 행렬곱해 output_layer에 저장해준다
    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            output_layer[i] += (w3[i][j]*hidden_output2[j]);
        }
    }
    //// ReLU연산을 한 뒤 output_output에 저장해준다
    for(i=0;i<output_size;i++)
    {
        output_output[i]=relu(output_layer[i]);
    }
}

void BP(int target)
{
    int i, j;
    for (i=0;i<output_size;i++)
    {
        if(i==target)
        {
            output_error[i]=1-output_output[i];
        }
        else
        {
            output_error[i]=0-output_output[i];
        }
    }

    //// 가중치 전치행렬 ////
    float tw1[input_size][hidden_size]={0,};
    float tw2[hidden_size][hidden_size]={0,};
    float tw3[hidden_size][output_size]={0,};

    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<input_size;j++)
        {
            tw1[j][i]=w1[i][j];
        }
    }

    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<input_size;j++)
        {
            tw2[j][i]=w2[i][j];
        }
    }

    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            tw3[j][i]=w3[i][j];
        }
    }

    //// hidden_layer들에 에러값 전달하기
    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<output_size;j++)
        {
            hidden_error2[i] = tw3[i][j]*output_error[j];
        }
    }

    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            hidden_error1[i] = tw2[i][j]*hidden_error2[j];
        }
    }
}

void update()
{
    int i,j;
    float de_relu;
    float dw1[hidden_size][input_size]={0,};
    float dw2[hidden_size][hidden_size]={0,};
    float dw3[output_size][hidden_size]={0,};

    //// w3 가중치 업데이트
    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            de_relu = (output_layer[i]>0)*output_error[i];
            dw3[i][j]=learning_rate * de_relu * hidden_output2[j];
            w3[i][j] = w3[i][j] + dw3[i][j];
        }
    }

    //// w2 가중치 업데이트
    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            de_relu = (hidden_layer2[i]>0)*hidden_error2[i];
            dw2[i][j]=learning_rate * de_relu * hidden_output1[j];
            w2[i][j] = w2[i][j] + dw2[i][j];
        }
    }

    //// w1 가중치 업데이트
    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<input_size;j++)
        {
            de_relu = (hidden_layer1[i]>0)*hidden_error1[i];
            dw1[i][j]=learning_rate * de_relu * input_layer[j];
            w1[i][j] = w1[i][j] + dw1[i][j];
        }
    }
}

int main()
{
    wreset();
    int i, j, k;
    
    ////학습
    for(i=0;i<learning_cycle;i++)
    {
        for(j=0;j<output_size;j++)
        {
            for(k=0;k<data_set;k++)
            {
                //// 이미지 csv파일을 불러와 가공해서 img 배열에 넣어주기
                FP();
                BP(j);
                update();
            }
        }
    }

    //// 추론 
    for(i=0;i<output_size;i++)
    {
        for(j=0;j<test_set;j++)
        {
            //// 이미지 csv파일을 불러와 가공해서 img 행렬에 넣어주기
            FP();
            //// 가장 큰 값을 가지는 인덱스 번호를 찾는다
            int index=0;
            int max = output_output[0];
            for(k=0;k<output_size;k++)
            {
                if(max<output_output[k])
                {
                    max=output_output[k];
                    index = k;
                }
            }
            // 인덱스 번호를 이용해 print해준다 
        }
    }

}
