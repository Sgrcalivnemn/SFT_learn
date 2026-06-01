


import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 1. 加载预测结果文件（请修改为你的实际路径）
file_path = "/root/autodl-tmp/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/eval_2026-05-31-21-19-02/generated_predictions.jsonl"  # 请替换为你的实际路径

predictions = []
true_labels = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        pred = item['predict'].strip()
        label = item['label'].strip()
        predictions.append(pred)
        true_labels.append(label)

# 2. 计算准确率
accuracy = accuracy_score(true_labels, predictions)
print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy:.2%})")

# 3. 计算精确率、召回率、F1（按正向/负向分别计算）
labels = ["正向", "负向"]
precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, labels=labels)

print("\n分类性能详情：")
for i, label in enumerate(labels):
    print(f"{label}:")
    print(f"  精确率 (Precision): {precision[i]:.4f}")
    print(f"  召回率 (Recall): {recall[i]:.4f}")
    print(f"  F1分数 (F1-score): {f1[i]:.4f}")
    print(f"  支持数 (Support): {support[i]}")

# 4. 混淆矩阵
cm = confusion_matrix(true_labels, predictions, labels=labels)
print("\n混淆矩阵：")
print(f"             预测正向    预测负向")
print(f"真实正向    {cm[0][0]:<12} {cm[0][1]:<12}")
print(f"真实负向    {cm[1][0]:<12} {cm[1][1]:<12}")

# 5. 统计总样本数和错误数
total = len(true_labels)
errors = total - (cm[0][0] + cm[1][1])
print(f"\n总样本数: {total}")
print(f"正确预测: {total - errors}")
print(f"错误预测: {errors}")