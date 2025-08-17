### **VQ-LLM 프로젝트 기획서 (PRD) - v1.2**

**1. 프로젝트 개요**

*   **1.1. 목적**
    *   새로운 VQ-LLM 프로젝트는 대규모 언어 모델(LLM)의 벡터 양자화(Vector Quantization, VQ)와 스칼라 양자화(Scalar Quantization)를 통해 추론 효율성을 극대화하고, 아웃라이어 완화 알고리즘을 통합하여 저비트(1-4비트) 양자화에서도 높은 정확도를 유지하는 오픈소스 프레임워크를 개발합니다. 기존 VQ-LLM의 코드북 캐시(codebook cache)와 연산 엔진(computation engine)을 개선하고, QLLM의 GPTQ 지원 및 VPTQ의 극저비트 성능을 결합하며, QuaRot의 회전 기반 아웃라이어 제거를 추가하여 메모리 사용량과 지연 시간을 최소화합니다.

*   **1.2. 배경**
    *   **VQ-LLM**: 코드북 캐시와 연산 엔진으로 VQ 통합 커널(GeMM, Attention 등)의 지연 시간을 64.36%~99.1% 감소. CUDA 기반 GPU 최적화 (HPCA 2025 논문).
    *   **QLLM**: GPTQ, AWQ, HQQ, VPTQ를 지원하는 툴박스로, ONNX 내보내기와 CLI/API 제공.
    *   **VPTQ**: 1-2비트 벡터 양자화로 405B 모델을 17시간 내 양자화, 메모리 19.54GB, AvgQA 68.6 (EMNLP 2024, arXiv:2409.17066).
    *   **QuaRot**: Hadamard 회전으로 아웃라이어 제거, 4비트 end-to-end 양자화, perplexity 손실 0.29 이하 (NeurIPS 2024, arXiv:2404.00456).
    *   **문제점**: 기존 VQ-LLM은 아웃라이어로 인한 코드북 왜곡 문제 미해결, QLLM은 저비트 특화 부족, VPTQ는 ONNX 미지원.
    *   **목표**: 아웃라이어 완화(QuaRot/SmoothQuant)와 VQ-LLM의 효율성을 통합하여 1-4비트 양자화에서 perplexity 손실 10-20% 감소, 메모리 30% 절감.

**2. 요구사항**

*   **2.1. 기능적 요구사항**
    1.  **핵심 양자화 엔진**:
        *   **VPTQ**: 열(column-wise) 방향 벡터 양자화 지원. 코드북 크기 동적 조정.
        *   **GPTQ**: 표준적인 스칼라 양자화 기법 지원.
        *   각 알고리즘에 최적화된 CUDA 커널 통합.
    2.  **아웃라이어 완화 모듈**:
        *   QuaRot의 Hadamard 회전 적용: 가중치(\( W \))와 활성화(\( X \))에 \( W' = RW \), \( X' = RX \).
        *   SmoothQuant의 활성화-가중치 이동: \( W' = sW \), \( X' = X/s \).
        *   CLI/API에서 전처리 옵션으로 선택 가능 (예: `--preprocess=quarot`).
    3.  **ONNX 내보내기**:
        *   QLLM 기반 ONNX 런타임 추론 지원.
        *   4비트 이하 모델의 ONNX 호환성 보장.
    4.  **사용자 인터페이스 및 경험**:
        *   Gradio 웹 앱으로 양자화/추론 시각화.
        *   명확한 CLI 및 Python API 제공.
        *   Jupyter Notebook 예제 제공.
    5.  **Hugging Face 통합**:
        *   Transformers v4.48.0+ 호환.
        *   양자화 모델을 HF Hub에 업로드 가능.

*   **2.2. 비기능적 요구사항**
    *   **성능**: LLaMA-70B에서 perplexity 손실 <0.3, zero-shot 정확도 98% 이상. 405B 모델 양자화 시간 <20시간, 메모리 사용량 <20GB. 추론 처리량(tok/s) 10 이상.
    *   **호환성**: NVIDIA GPU(CUDA 11.8+), Python 3.10+, PyTorch 2.0+.
    *   **확장성**: 7B~405B 모델 지원.
    *   **라이선스**: MIT 라이선스, 오픈소스 배포.

**3. 프로젝트 범위**

*   **3.1. 포함 범위**:
    *   **VPTQ**와 **GPTQ** 양자화 알고리즘 구현 및 통합.
    *   QuaRot 및 SmoothQuant 아웃라이어 완화 기법 통합.
    *   VQ-LLM의 코드북 캐시 및 연산 엔진 개념을 참고하여 성능 최적화.
    *   Gradio UI 및 ONNX 내보내기 기능.
    *   LLaMA, Mistral, Deepseek 모델 초기 지원.
*   **3.2. 제외 범위**:
    *   **초기 버전 제외**: AWQ, HQQ 등 다른 양자화 기법 (향후 확장 고려).
    *   재훈련 기반 양자화(QAT).
    *   비-NVIDIA GPU(예: AMD ROCm) 지원.
    *   1비트 미만 양자화.

**4. 기술 설계**

*   **4.1. 시스템 아키텍처**:
    *   **입력 처리**: Hugging Face 모델 로드(Transformers API). CLI 또는 API로 양자화 설정(bits, method, preprocess) 입력.
    *   **아웃라이어 완화 모듈**: (선택 사항) QuaRot 또는 SmoothQuant 전처리 적용.
    *   **양자화 엔진**: 설정에 따라 VPTQ 또는 GPTQ 알고리즘 실행. **핵심 양자화 로직은 가독성과 확장성을 위해 Python으로 구현**하며, **성능이 중요한 추론 커널은 C++/CUDA를 사용**한다.
    *   **출력**: 양자화된 모델 저장(HF Hub 또는 로컬), ONNX 모델로 내보내기.

*   **4.2. 코드 구조**:
    *   Python으로 작성된 상위 수준의 API 및 양자화 알고리즘과, C++/CUDA로 작성된 고성능 추론 커널을 분리하여 관리한다.
    ```
    vqllm/
    ├── src/vqllm/
    │   ├── api.py              # [Python] 사용자용 고수준 API (AutoQuantizer)
    │   ├── cli.py              # [Python] 커맨드 라인 인터페이스
    │   ├── quantization/       # [Python] 양자화 알고리즘
    │   │   ├── base.py         # Quantizer 기본 클래스
    │   │   ├── gptq.py         # GPTQ 로직
    │   │   └── vptq/           # VPTQ 로직
    │   ├── layers/             # [Python] 양자화 레이어 (CUDA 커널 호출)
    │   │   ├── base_layer.py   # Quantized Layer 기본 클래스
    │   │   ├── gptq_linear.py
    │   │   └── vq_linear.py
    │   ├── outlier/            # [Python] 아웃라이어 완화 모듈
    │   │   ├── quarot.py
    │   │   └── smoothquant.py
    │   └── utils/              # [Python] 공통 유틸리티
    ├── csrc/                   # [C++/CUDA] 고성능 추론 커널
    │   ├── gptq/
    │   └── vptq/
    ├── examples/
    └── tests/
    ```

*   **4.3. 양자화 처리 방향**:
    *   **초기 버전**: **열(Column-wise)** 방향의 벡터/스칼라 양자화를 우선 지원합니다. 이는 분석된 `VPTQ`와 `GPTQ`의 핵심 구현 방식이며 안정성이 검증되었습니다.
    *   **향후 로드맵**: **행(Row-wise)** 방향 양자화 지원을 주요 마일스톤으로 계획합니다. 이는 오차 보상 알고리즘의 수정이 필요하므로, 초기 버전 출시 후 아키텍처 안정성을 확보한 뒤 추가 개발을 진행합니다.

*   **4.4. 의존성**:
    *   Python 3.10+, PyTorch 2.0+, Transformers 4.48.0+.
    *   CUDA 11.8+, NumPy, SciPy, Gradio, Datasets, Safetensors.

**5. 개발 계획**

*   **5.1. 마일스톤**:
    1.  **1개월**: 프로젝트 구조 설정, 기존 `VPTQ` 및 `QLLM`의 GPTQ 코드 분석 및 `vqllm`으로 이관 시작.
    2.  **2-3개월**: VPTQ 및 GPTQ 핵심 로직 통합 완료. 아웃라이어 완화 모듈(QuaRot, SmoothQuant) 프로토타입 구현.
    3.  **4-5개월**: ONNX 내보내기 기능 구현 및 테스트. Gradio UI 개발.
    4.  **6개월**: LLaMA-7B/70B 모델 대상 전체 파이프라인 테스트 및 성능 벤치마크.
    5.  **7-8개월**: 405B 대형 모델 테스트, HF Hub 배포 준비, 문서화 및 예제 작성.

*   **5.2. 리소스**:
    *   **인력**: ML 엔지니어 2명, CUDA 개발자 1명, UI 개발자 1명.
    *   **하드웨어**: NVIDIA A100 GPU 2대, 128GB RAM 서버.
    *   **예산**: 오픈소스 프로젝트로 최소화, 클라우드 비용 $5,000 추정.

**6. 성공 기준**

*   **정량적**:
    *   Perplexity 손실: LLaMA-70B@2비트(VPTQ)에서 <0.3.
    *   메모리 사용량: 405B 모델 <20GB.
    *   양자화 시간: 405B <20시간.
    *   추론 처리량: tok/s >10.
*   **정성적**:
    *   Gradio UI를 통해 사용자 친화적인 양자화/추론 환경 제공.
    *   HF Hub에서 100+ 다운로드, GitHub 스타 500+ 달성.

**7. 리스크 및 완화 전략**

*   **리스크 1**: 아웃라이어 완화 기법과 벡터 양자화 통합 시 예상치 못한 성능 저하 발생.
    *   **완화**: 각 모듈(회전, 스케일링, 양자화)을 독립적으로 테스트하고, 다양한 데이터셋에서 성능을 교차 검증하여 최적의 조합을 찾습니다.
*   **리스크 2**: CUDA 커널 최적화의 복잡성.
    *   **완화**: `VPTQ`와 `QLLM`의 기존 커널을 최대한 재사용하고, 초기에는 Python/Triton으로 프로토타이핑 후 점진적으로 최적화합니다.
*   **리스크 3**: ONNX 호환성 문제.
    *   **완화**: `QLLM`의 검증된 ONNX 내보내기 모듈을 기반으로, 커스텀 양자화 연산자에 대한 ONNX 지원을 단계적으로 테스트하고 강화합니다.

**8. 참고 자료**

*   VQ-LLM 논문 (HPCA 2025).
*   VPTQ 논문 (EMNLP 2024, arXiv:2409.17066).
*   QuaRot 논문 (NeurIPS 2024, arXiv:2404.00456).
*   QLLM GitHub: [https://github.com/wejoncy/QLLM](https://github.com/wejoncy/QLLM)
*   VPTQ GitHub: [https://github.com/microsoft/VPTQ](https://github.com/microsoft/VPTQ)