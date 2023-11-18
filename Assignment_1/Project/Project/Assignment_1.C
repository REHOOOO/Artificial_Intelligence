#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//// 파라미터 ////
#define input_size 16*16
#define output_size 7
#define hidden_size 1000
int learning_cycle = 100;
int data_set=10;
int test_set = 20;
double learning_rate = 0.0000000000001;

///// 신경망 구현 ////////
double input_layer[input_size]={0,};
double output_layer[output_size]={0,};
double hidden_layer1[hidden_size]={0,};
double hidden_layer2[hidden_size]={0,};

double w1[hidden_size][input_size]={0,};
double w2[hidden_size][hidden_size]={0,};
double w3[output_size][hidden_size]={0,};

double hidden_output1[hidden_size]={0,};
double hidden_output2[hidden_size]={0,};
double output_output[output_size]={0,};

double output_error[output_size]={0,};
double hidden_error1[hidden_size]={0,};
double hidden_error2[hidden_size]={0,};

double img[input_size]={0,};

//// 가중치 전치행렬 ////
double tw1[input_size][hidden_size] = { 0, };
double tw2[hidden_size][hidden_size] = { 0, };
double tw3[hidden_size][output_size] = { 0, };

//// 가중치 업데이트 ////
double dw1[hidden_size][input_size] = { 0, };
double dw2[hidden_size][hidden_size] = { 0, };
double dw3[output_size][hidden_size] = { 0, };


//// 가중치 초기화 ////
void wreset()
{
    int i, j;

    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<input_size;j++)
        {
            w1[i][j]=rand()/(double)RAND_MAX;
        }
    }

    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            w2[i][j]=rand()/(double)RAND_MAX;
        }
    }

    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            w3[i][j]=rand()/(double)RAND_MAX;
        }
    }
}

//2차원 배열 크기 알아내는법
//int row_A = sizeof(A) / sizeof(A[0]);
//int col_A = sizeof(A[0])/sizeof(float);

void img_input(alphabet, num, check)
{
    int i = 0;
    char img_path[30];
    char alpha[5] = { alphabet + 't' }; // 읽어올 파일의 이름 중 알파벳 부분을 만들어준다 
    char n[5];  // 읽어올 파일의 이름 중 숫자 부분을 만들어준다 
    num++;
    if (num >= 10)
    {
        n[0] = num / 10 + '0';
        n[1] = num % 10 + '0';
        n[2] = NULL;
    }
    else
    {
        n[0] = num + '0';
        n[1] = NULL;
    }
    char c[5] = ".csv";
    
    char str[6000];
    char* p;
    if (check == 0)
    {
        strcpy(img_path, "image/learning/");
        strcat(img_path, alpha);
        strcat(img_path, n);
        strcat(img_path, c);
    }
    else
    {
        strcpy(img_path, "image/input/");
        strcat(img_path, alpha);
        strcat(img_path, n);
        strcat(img_path, c);
    }

    FILE* pFile = NULL;
    pFile = fopen(img_path, "r");
    if (pFile == NULL)
    {
        printf("no file");
        return 0;
    }
    
    fgets(str, 7000, pFile);
    fclose(pFile);

    p = strtok(str, ",");
    for (i = 0; i < input_size;i++)
    {
        img[i] = atof(p); // 문자열을 실수로 바꾸어준다 
        p = strtok(NULL, ",");
    }

}

double relu(double x)
{
    double o = max(0,x);
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
    for (i = 0; i < hidden_size; i++) // hidden_layer1 초기화
    {
        hidden_layer1[i] = 0;
    }
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
    for (i = 0; i < hidden_size; i++) // hidden_layer2 초기화
    {
        hidden_layer2[i] = 0;
    }
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
    for (i = 0; i < output_size; i++) // hidden_layer1 초기화
    {
        output_layer[i] = 0;
    }
    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            output_layer[i] += (w3[i][j] * hidden_output2[j]);
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
            output_error[i]=1000000-output_output[i];
        }
        else
        {
            output_error[i]= 100000 -output_output[i];
        }
    }

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
    double de_relu;

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
    printf("시작\n");
    wreset();
    int i, j, k;
    
    ////학습
    for(i=0;i<learning_cycle;i++)
    {
        printf("%d\n", i);
        for(j=0;j<output_size;j++)
        {
            for(k=0;k<data_set;k++)
            {
                img_input(j, k, 0);
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
            img_input(i, j, 1);
            FP();
            //// 가장 큰 값을 가지는 인덱스 번호를 찾는다
            int index=0;
            double max = output_output[0];
            for(k=0;k<output_size;k++)
            {
                printf("%f ", output_output[k]);
                if(max<output_output[k])
                {
                    max=output_output[k];
                    index = k;
                }
            }
            printf("\n");

            char file_name[50] = "파일명: ";
            char alpha[5] = { i + 't' };
            char n[5];
            if (j >= 10)
            {
                n[0] = j / 10 + '0';
                n[1] = j % 10 + '0';
                n[2] = NULL;
            }
            else
            {
                n[0] = j + '0';
                n[1] = NULL;
            }
            char infer[10] = " 추론값: ";
            char result[5] = { index + 't' };
            strcat(file_name, alpha);
            strcat(file_name, n);
            strcat(file_name, infer);
            strcat(file_name, result);
            printf("%s\n", file_name);

        }
    }

}
