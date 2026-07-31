# credit_model_037 / champion_v1 模型卡

## 身份

- 模型角色：Champion
- 算法族：CatBoost
- 特征策略：F01
- 随机种子：2026037
- 生命周期状态：CHAMPION_V1_W1_HEALTHY
- 生产就绪：否

## 用途与非用途

用于任务四比赛演示中的独立在役信贷风险模型评分和监测。输出为坏样本的
`calibrated_pd`。不得作为真实银行拒贷阈值，不代表已完成 W4 OOT、合规、
公平性、策略或生产审批。

## 训练方法

候选只在 W0_train 拟合并在 W0_tune 选择；最终模型仅使用
W0_train + W0_tune 重拟合。Platt 与 Isotonic 只在 W0_calibration_fit
拟合，并只在 W0_calibration_select 以 Brier、ECE 选择。冻结阈值只在
W0_threshold 上选择。W1 只做健康确认，W2/W3 只做监测。

最终参数：`{"auto_class_weights": null, "depth": 4, "iterations": 500, "l2_leaf_reg": 3, "learning_rate": 0.03, "random_strength": 0.5}`

校准器：`Isotonic`

阈值：`0.06976744186046512`，分数空间 `calibrated_pd`，比较符 `>=`。

## W1 健康指标

- ROC-AUC：0.98107552
- KS：0.87365747
- Bad Recall：0.91618311
- Brier：0.01087227
- ECE：0.00334193

## 数据与限制

开发数据 SHA-256：`b5089a4ff7e2b1b4bc392fd847b77429c918fa1989c57b43aa4a39431bf786c5`。`id_card`、原始 `apply_time` 和
`is_bad` 不进入模型矩阵。敏感字段可能存在于部分特征策略中，必须结合
F06 治理对照和独立公平性审计解读。任何 W1–W4 结果均不得反向用于调参。
