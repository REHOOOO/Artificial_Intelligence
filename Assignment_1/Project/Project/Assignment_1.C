#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//// 파라미터 ////
#define input_size 16*16 // 16 * 16 픽셀
#define output_size 7 // t, u, v, w, x, y, z 7개 
#define hidden_size 1000 // 히든 레이어 크기
int learning_cycle = 100; // 학습 사이클 횟수 
int data_set=10; // 데이터 셋의 개수 
int test_set = 20; // 추론시 사용하는 데이터 셋의 개수 
double learning_rate = 0.000000000000000000001; // 학습률

/////// 신경망 구현 ////////
//// 레이어 ////
double input_layer[input_size]={0,};
double output_layer[output_size]={0,};
double hidden_layer1[hidden_size]={0,};
double hidden_layer2[hidden_size]={0,};
double hidden_layer3[hidden_size] = { 0, };

//// 가중치 ////
double w1[hidden_size][input_size]={0,};
double w2[hidden_size][hidden_size]={0,};
double w3[hidden_size][hidden_size]={0,};
double w4[output_size][hidden_size] = { 0, };

/// 레이어 출력값 ///
double hidden_output1[hidden_size]={0,};
double hidden_output2[hidden_size]={0,};
double hidden_output3[hidden_size] = { 0, };
double output_output[output_size]={0,};

//// 에러값 ////
double output_error[output_size]={0,};
double hidden_error1[hidden_size]={0,};
double hidden_error2[hidden_size]={0,};
double hidden_error3[hidden_size] = { 0, };

//// 입력 이미지 ////
double img[input_size]={0,};

//// 가중치 전치행렬 ////
double tw1[input_size][hidden_size] = { 0, };
double tw2[hidden_size][hidden_size] = { 0, };
double tw3[hidden_size][hidden_size] = { 0, };
double tw4[hidden_size][output_size] = { 0, };

//// 가중치 업데이트 ////
double dw1[hidden_size][input_size] = { 0, };
double dw2[hidden_size][hidden_size] = { 0, };
double dw3[hidden_size][hidden_size] = { 0, };
double dw4[output_size][hidden_size] = { 0, };


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

    for (i = 0; i < hidden_size; i++)
    {
        for (j = 0; j < hidden_size; j++)
        {
            w3[i][j] = rand() / (double)RAND_MAX;
        }
    }

    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            w4[i][j]=rand()/(double)RAND_MAX;
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
    if (num >= 10) // num이 10보다 크거나 같다면 (여기서는 100 미만의 값이 들어온다고 가정)
    {
        n[0] = num / 10 + '0'; // n[0]에는 10으로 나눈값에 0의 아스키코드값을 더해 저장
        n[1] = num % 10 + '0'; // n[1]에는 10으로 나눈 나머지값에 0의 아스키코드값을 더해 저장
        n[2] = NULL; // 문자열의 끝
    }
    else // num이 10보다 작다면
    {
        n[0] = num + '0'; // n[0]은 num에 0의 아스키코드값을 더해 저장
        n[1] = NULL; // 문자열의 끝 
    }
    char c[5] = ".csv";
    
    char str[7000];
    char* p;
    if (check == 0) // check를 통해 learning 폴더에서 읽어올지 input 폴더에서 읽어올지 결정
    {
        strcpy(img_path, "image/learning/");
        strcat(img_path, alpha);
        strcat(img_path, n);
        strcat(img_path, c); // 앞에서 만들었던 문자열들을 합쳐준다
    }
    else
    {
        strcpy(img_path, "image/input/");
        strcat(img_path, alpha);
        strcat(img_path, n);
        strcat(img_path, c); // 앞에서 만들었던 문자열들을 합쳐준다
    }

    FILE* pFile = NULL;
    pFile = fopen(img_path, "r"); // 앞에서 만든 img_path에 있는 파일을 읽기 모드로 불러온다
    if (pFile == NULL) // 만약 읽을 파일이 없다면 
    {
        printf("no file");
        return 0;
    }
    
    fgets(str, 7000, pFile); // str 배열에 읽을 파일을 넣어준다
    fclose(pFile);

    p = strtok(str, ","); // , 문자를 기준으로 문자열을 잘라준다 
    for (i = 0; i < input_size;i++) // input_size만큼 for문을 돌면서 
    {
        img[i] = atof(p); // 문자열을 실수로 바꾸어 img에 저장해준다 
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

    //// hidden_layer3 ////
    //// w3와 hidden_output2을 행렬곱해 hidden_layer3에 저장해준다
    for (i = 0; i < hidden_size; i++) // hidden_layer3 초기화
    {
        hidden_layer3[i] = 0;
    }
    for (i = 0; i < hidden_size; i++)
    {
        for (j = 0; j < hidden_size; j++)
        {
            hidden_layer3[i] += (w3[i][j] * hidden_output2[j]);
        }
    }
    //// ReLU연산을 한 뒤 hidden_output2에 저장해준다
    for (i = 0; i < hidden_size; i++)
    {
        hidden_output3[i] = relu(hidden_layer3[i]);
    }

    //// output_layer ////
    //// w4과 hidden_output3을 행렬곱해 output_layer에 저장해준다
    for (i = 0; i < output_size; i++) // output_layer 초기화
    {
        output_layer[i] = 0;
    }
    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            output_layer[i] += (w4[i][j] * hidden_output3[j]);
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
        if(i==target) // i가 target과 같다면 목표값을 1로 설정해준다 
        {
            output_error[i]=1-output_output[i]; // 오차 = 목표값 - 결과값
        }
        else // target과 다르다면 목표값을 0으로 설정해준다 
        {
            output_error[i]= 0 -output_output[i]; // 오차 = 목표값 - 결과값
        }
    }

    // w1 전치행렬
    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<input_size;j++)
        {
            tw1[j][i]=w1[i][j];
        }
    }

    // w2 전치행렬
    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            tw2[j][i]=w2[i][j];
        }
    }

    // w3 전치행렬
    for (i = 0; i < hidden_size; i++)
    {
        for (j = 0; j < hidden_size; j++)
        {
            tw3[j][i] = w3[i][j];
        }
    }

    // w4 전치행렬
    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            tw4[j][i]=w4[i][j];
        }
    }

    //// hidden_layer들에 에러값 전달하기
    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<output_size;j++)
        {
            hidden_error3[i] = tw4[i][j]*output_error[j];
        }
    }

    for(i=0;i<hidden_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            hidden_error2[i] = tw3[i][j]*hidden_error3[j];
        }
    }

    for (i = 0; i < hidden_size; i++)
    {
        for (j = 0; j < hidden_size; j++)
        {
            hidden_error1[i] = tw2[i][j] * hidden_error2[j];
        }
    }
}

void update()
{
    int i,j;
    double de_relu;

    //// w4 가중치 업데이트
    for(i=0;i<output_size;i++)
    {
        for(j=0;j<hidden_size;j++)
        {
            de_relu = (output_layer[i]>0)*output_error[i];
            dw4[i][j]=learning_rate * de_relu * hidden_output3[j];
            w4[i][j] = w4[i][j] + dw4[i][j];
        }
    }

    //// w3 가중치 업데이트
    for (i = 0; i < hidden_size; i++)
    {
        for (j = 0; j < hidden_size; j++)
        {
            de_relu = (hidden_layer3[i] > 0) * hidden_error3[i];
            dw3[i][j] = learning_rate * de_relu * hidden_output2[j];
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
    for(i=0;i<learning_cycle;i++) // 학습 사이클
    {
        printf("%d\n", i); // cycle 수를 출력
        for(j=0;j<output_size;j++) // t~z까지 
        {
            for(k=0;k<data_set;k++) // 데이터셋의 개수만큼 반복
            {
                img_input(j, k, 0);
                FP();
                BP(j);
                update();
            }
        }
    }

    //// 추론 
    for(i=0;i<output_size;i++) // t~z 까지 
    {
        for(j=0;j<test_set;j++) // 데이터셋의 개수만큼 반복
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
            char infer[20] = " 추론값: ";
            char result[5] = { index + 't' };
            strcat(file_name, alpha);
            strcat(file_name, n);
            strcat(file_name, infer);
            strcat(file_name, result);
            printf("%s\n", file_name); // 파일명: (파일명) 추론값: (추론값) 형태로 출력된다 

        }
    }

}
