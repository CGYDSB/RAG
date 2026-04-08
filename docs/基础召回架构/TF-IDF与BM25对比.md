# TF-IDF 与 BM25 对比

当前项目关键词检索支持两种评分方式，通过 `settings.yaml` 的 `keyword_method` 参数切换。

---

## 一、公式对比

### TF-IDF

$$\text{score}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

$$\text{IDF}(t) = \ln\frac{N + 1}{df(t) + 1}$$

- $N$：文档总数
- $df(t)$：包含词 $t$ 的文档数
- $\text{TF}(t, d)$：词 $t$ 在文档 $d$ 中的出现次数

词频越高，分数线性增长，没有上限。

### BM25

$$\text{score}(t, d) = \text{IDF}(t) \times \frac{\text{TF}(t,d) \cdot (k_1 + 1)}{\text{TF}(t,d) + k_1 \cdot \left(1 - b + b \cdot \dfrac{|d|}{\text{avgdl}}\right)}$$

$$\text{IDF}(t) = \ln\left(\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1\right)$$

- $k_1 = 1.5$：TF 饱和系数，控制词频的影响上限
- $b = 0.75$：文档长度归一化系数（0=不归一化，1=完全归一化）
- $|d|$：文档 $d$ 的 token 数
- $\text{avgdl}$：所有文档的平均 token 数

---

## 二、核心差异

### 差异一：TF 饱和

**TF-IDF**：词频线性增长，一个词出现 100 次的得分是出现 10 次的 10 倍。

**BM25**：词频有饱和上限。当 $k_1 = 1.5$ 时，TF 趋向无穷大时，分子趋向 $(k_1 + 1) = 2.5$，分母趋向 TF，整体趋向 2.5。

```
TF=1:  BM25 TF项 = 1×2.5 / (1 + 1.5) = 1.0
TF=5:  BM25 TF项 = 5×2.5 / (5 + 1.5) = 1.92
TF=10: BM25 TF项 = 10×2.5 / (10 + 1.5) = 2.17
TF=50: BM25 TF项 = 50×2.5 / (50 + 1.5) = 2.42
TF=∞:  BM25 TF项 → 2.5（上限）
```

**意义**：一个词在文档里出现 50 次和 10 次，BM25 认为差别不大（2.42 vs 2.17）；TF-IDF 认为前者是后者的 5 倍。对于技术文档，某个术语反复出现不代表文档"更相关"，BM25 的饱和更合理。

### 差异二：文档长度归一化

**TF-IDF**：不考虑文档长度。一篇 5000 字的文档里"函数"出现 20 次，和一篇 500 字的文档里出现 20 次，得分相同。

**BM25**：按文档长度归一化。长文档里词频高是"自然的"，不应该得到额外奖励。

```
短文档（500字，avgdl=1000）：
  长度因子 = 1 - 0.75 + 0.75 × (500/1000) = 0.625
  → 分母更小，得分更高（短文档里出现同样次数，相关性更强）

长文档（2000字，avgdl=1000）：
  长度因子 = 1 - 0.75 + 0.75 × (2000/1000) = 1.75
  → 分母更大，得分更低（长文档里出现同样次数，相关性被打折）
```

**意义**：当前项目 chunk_size=256，chunk 长度差异不大，这个差异影响有限。但如果文档长度差异大（如混合了短摘要和长报告），BM25 的归一化会更公平。

### 差异三：IDF 公式

| | TF-IDF | BM25 |
|---|---|---|
| 公式 | $\ln\frac{N+1}{df+1}$ | $\ln\left(\frac{N-df+0.5}{df+0.5}+1\right)$ |
| 极端情况 | $df=N$ 时 IDF=0 | $df=N$ 时 IDF≈ln(1)=0，但更平滑 |
| 负值 | 不会出现（加了平滑） | 不会出现（+1 保证正值） |

差异较小，主要是平滑方式不同。

---

## 三、当前项目实现

```python
# src/retriever.py — _keyword_search()

if self.keyword_method == "bm25":
    k1, b = 1.5, 0.75
    for token in tokens:
        df = len(self._keyword_index[token])
        idf = np.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)
        for doc_id, tf in self._keyword_index[token].items():
            dl = self._doc_lengths.get(doc_id, self._avg_doc_length)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avg_doc_length))
            scores[doc_id] += idf * tf_norm
else:
    # TF-IDF
    for token in tokens:
        idf = np.log((self._total_docs + 1) / (len(self._keyword_index[token]) + 1))
        for doc_id, freq in self._keyword_index[token].items():
            scores[doc_id] += freq * idf
```

BM25 需要在索引时额外记录每个文档的 token 数（`_doc_lengths`）和平均文档长度（`_avg_doc_length`），这两个值在 `build_keyword_index()` 时计算并持久化。

---

## 四、如何切换

```yaml
# config/settings.yaml
retrieval:
  hybrid_search:
    keyword_method: "bm25"    # BM25（默认，推荐）
    # keyword_method: "tfidf" # 切换回 TF-IDF
```

切换后需要重新索引（BM25 需要文档长度信息，旧索引没有这个字段）：

```bash
python main.py index --source data/raw/ --clear
```

---

## 五、选择建议

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 文档长度差异大（混合短摘要和长报告） | BM25 | 长度归一化更公平 |
| 专有名词密集（型号、药品名反复出现） | BM25 | TF 饱和避免高频词过度主导 |
| chunk 长度均匀（当前项目 chunk_size=256） | 两者差异不大 | BM25 略优 |
| 调试/快速验证 | TF-IDF | 实现更简单，便于理解 |

当前项目默认使用 BM25，对于技术文档场景（专有名词密集、chunk 长度相对均匀）是更合适的选择。
