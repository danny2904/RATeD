3. Rationale-quality Trigger

Tính một chỉ số nội bộ cho chất lượng rationale, ví dụ độ tập trung của attention hoặc độ bao phủ span. Nếu rationale phân tán hoặc quá ngắn, chuyển sang Stage-2. Cách này trực tiếp tối ưu explainability.

4. Cost-aware Adaptive Routing

Xây dựng một meta-classifier nhỏ học quyết định có nên gọi LLM hay không dựa trên đặc trưng như confidence, entropy, độ dài câu, số token nghi vấn. Cơ chế này học được “sweet spot” thay vì đặt ngưỡng cố định.

Learned gating network: một classifier nhỏ nhận các đặc trưng sẵn có từ backbone như entropy, margin top-2, độ dài câu, số identity terms, độ tập trung rationale, rồi dự đoán “route or not”. Huấn luyện bằng nhãn mục tiêu là việc Stage-2 có cải thiện đúng hay không trên dev set.

Cost-aware objective: tối ưu 
Utility
=
Δ
Metric
−
𝜆
⋅
Cost
Utility=ΔMetric−λ⋅Cost, trong đó cost là số lần gọi LLM hoặc latency, metric có thể là Macro-F1 hoặc một tổng hợp gồm Macro-F1 và IoU.