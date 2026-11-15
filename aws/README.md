# (선택) GPU 서버로 옮기기: EC2와 SageMaker

2일차 실습은 전부 아이맥 안에서 끝납니다. 이 문서는 **더 큰 모델을 스승으로 쓰고 싶을 때** AWS의 GPU를 빌리는 방법입니다. Amazon Bedrock은 쓰지 않습니다.

> ⚠️ 이 문서의 절차는 **강사가 실제 AWS 계정에서 실행해 보지 않았습니다.** 요금이 나가는 작업이라 확인 없이 돌리지 않았습니다. 요금과 AMI 이름은 바뀔 수 있으니 실행 전에 링크에서 확인하세요.

## 무엇이 달라지나

| | 아이맥 (기본) | GPU EC2 |
|---|---|---|
| 스승 모델 | `gemma4:e4b` (9.5GB) | `gemma3:27b` (17GB) 등 더 큰 모델 |
| 비용 | 0원 | `g5.xlarge` 기준 시간당 약 1달러, 하루 켜 두면 약 24달러 |
| 준비 시간 | 없음 | 15~20분 (서버 시작, 모델 내려받기) |
| 노트북 코드 | 그대로 | 그대로. 주소만 바꿈 |

노트북 6~8은 `OLLAMA_HOST` 환경변수의 주소로 Ollama를 부릅니다. 그래서 **코드는 한 줄도 안 고치고** 스승만 바꿀 수 있습니다. 제자 학습(노트북 9)은 계속 아이맥의 MLX로 합니다.

## 방법 1. GPU EC2에 Ollama 띄우기

**1) 사전 확인**: 새 AWS 계정은 GPU 인스턴스 한도가 0입니다. Service Quotas에서 "Running On-Demand G and VT instances"가 4 vCPU 이상인지 먼저 확인하고, 0이면 증가를 신청합니다(승인에 시간이 걸릴 수 있음).

**2) 인스턴스 시작** (콘솔 → EC2 → 인스턴스 시작)

- AMI: "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)" (NVIDIA 드라이버가 들어 있음)
- 인스턴스 유형: `g5.xlarge` (NVIDIA A10G 24GB)
- 저장 공간: 100GB
- 보안 그룹: **SSH(22)만, 내 IP에서만** 허용. Ollama 포트(11434)는 열지 않습니다

**3) 서버에서 Ollama 설치와 모델 내려받기**

```bash
ssh -i 내키.pem ubuntu@서버주소
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:27b
```

**4) 아이맥에서 SSH 터널로 연결** (새 터미널을 열고 켜 둡니다)

```bash
ssh -i 내키.pem -N -L 11435:localhost:11434 ubuntu@서버주소
```

서버의 Ollama가 내 맥의 `localhost:11435`로 보입니다. 포트를 인터넷에 열지 않으므로 다른 사람이 내 GPU를 쓸 수 없습니다.

**5) 주소를 바꿔 JupyterLab 실행**

```bash
cd mcp-lab
OLLAMA_HOST=http://localhost:11435 .venv/bin/python -m jupyter lab --notebook-dir=example
```

노트북 6~8의 설정 셀에서 `EXPERT_MODEL = "gemma3:27b"`로 바꾸고 실행합니다. 출력에 `Ollama 주소: http://localhost:11435`가 보이면 서버의 GPU를 쓰는 것입니다.

**6) 끝나면 반드시 인스턴스 종료(Terminate)**. 중지(Stop)만 하면 디스크 요금이 계속 나갑니다.

## 방법 2. SageMaker

참고한 AWS 블로그의 SK바이오팜 팀은 신약 전용 모델 TxGemma(`google/txgemma-2b-predict`)를 SageMaker 실시간 엔드포인트(`ml.g5.2xlarge`)에 올려 분자 특성을 예측했습니다. 오늘 실습에서 RDKit으로 계산한 자리(`molecule_properties`)가 그 부분입니다.

- 같은 구성을 하려면 SageMaker JumpStart나 Hugging Face 컨테이너로 엔드포인트를 만들고, `drug_tools.py`의 `molecule_properties`에서 그 엔드포인트를 부르도록 바꾸면 됩니다.
- 엔드포인트는 **켜져 있는 동안 계속 요금**이 나갑니다. 실습이 끝나면 엔드포인트를 삭제하세요.
- TxGemma는 Hugging Face에서 사용 조건에 동의해야 내려받을 수 있습니다.
- 제자 모델 학습을 SageMaker Training Job으로 옮길 수도 있지만, 이번 실습 크기(질문 400개, 3분)에서는 아이맥이 더 빠르고 쌉니다.

## 요금 확인

- EC2 온디맨드 요금: https://aws.amazon.com/ko/ec2/pricing/on-demand/
- SageMaker 요금: https://aws.amazon.com/ko/sagemaker/pricing/
- 요금 계산기: https://calculator.aws/
