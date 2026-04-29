# Poster Graph Draft

## 1. Motivation 图

这张图放在左上角，作用是让人一眼看懂你到底在解决什么问题。

### 你要表达的是

- uniform INT4 给所有层同样精度
- mixed policy 在同样总预算下重新分配精度
- 不是增加预算，而是改变预算分配方式

### 图里应该有什么

- 上面一行：`Uniform INT4`
- 下面一行：`Mixed Task-Aware`
- 上面全是一样大小的绿色 `INT4` 块
- 下面是红色 `INT8`、绿色 `INT4`、灰色 `INT2` 混合块
- 两行总长度完全一样
- 下方一条细括号或标尺写 `Same exact raw weight footprint`

### 生成 prompt

```text
Create a clean academic infographic panel comparing uniform INT4 and task-aware mixed precision under the same exact memory budget. Use two horizontal rows of identical total width. The top row is labeled "Uniform INT4" and contains evenly sized green blocks, all labeled INT4. The bottom row is labeled "Mixed Task-Aware" and contains a mixture of soft red INT8 blocks, muted green INT4 blocks, and cool light gray INT2 blocks with different widths, but the full row has exactly the same total width as the top row. Add a thin bracket or scale underneath both rows labeled "Same exact raw weight footprint". White background, flat vector scientific poster style, crimson accent, clean typography, minimal clutter, crisp and publication-ready.
```

## 2. 方法总流程图

这张图是整张 poster 最重要的一张方法图。  
它要回答：你们现在真正用的算法是什么。

### 你要表达的是

- sensitivity profiling
- rank by value/size
- build exact-budget frontier
- coarse search
- fine search
- final PTQ and evaluation

### 图里应该有什么

- 左到右 8 个步骤卡片
- 每个卡片有一个简洁图标
- 箭头连接
- 最后落到 final selected policy 和 evaluation

### 生成 prompt

```text
Create a wide left-to-right workflow diagram for a method called "Structured Exact-Budget Coarse-to-Fine Mixed-Precision Search". Show eight clean rounded cards connected by arrows. Card 1: target benchmark and calibration set. Card 2: sensitivity profiling over model groups with a heatmap icon. Card 3: rank groups by sensitivity divided by true size using a sorted bar chart icon. Card 4: build an exact-budget mixed frontier with INT2, INT4, and INT8 assignments. Card 5: coarse search across 8 sectors of the frontier. Card 6: fine search inside the best sector with 4 refined candidates. Card 7: select the final mixed-precision policy. Card 8: real PTQ artifact and final benchmark evaluation. Use white background, flat vector academic poster style, crimson accents, muted green for INT4, soft red for INT8, cool gray for INT2, minimal but readable labels, generous spacing, crisp scientific illustration.
```

## 3. Policy Frontier 图

这张图的作用是把“搜索空间”讲清楚。  
很多人会问：你到底搜的是啥。  
这张图要回答：我们搜的是一条有序 frontier，不是随机乱搜。

### 你要表达的是

- 所有 candidate 都满足同一个 exact INT4 budget
- 从左到右，INT8 + INT2 逐渐减少，INT4 逐渐增加
- 这是一条有序 policy spectrum

### 图里应该有什么

- 8 个柱子或 8 个横条，标成 `S1` 到 `S8`
- 每个柱子的总高度一样
- 左边红色和灰色更多，右边绿色更多
- 顶部或底部写 `Same budget across all candidates`

### 生成 prompt

```text
Create a scientific infographic showing a policy frontier under a fixed INT4 budget. Display eight candidate policies labeled S1 through S8. Each candidate should have the same total height or total width to indicate identical total budget. Inside each candidate, show a stacked composition of INT8 in soft red, INT4 in muted green, and INT2 in cool light gray. On the left side, candidates have more INT8 and INT2; moving to the right, the red and gray portions shrink and the green INT4 portion grows until the rightmost candidate is almost fully INT4. Add a subtle caption saying "Same exact raw weight budget" and a direction arrow from "more INT8 + INT2" to "more uniform INT4". White background, flat vector academic poster style, very clean and interpretable.
```

## 4. Coarse-to-Fine Search 图

这张图建议做成和你截图那种风格最像的版本。  
因为它最适合用卡片式视觉表达。

### 你要表达的是

- 第一轮把 frontier 分成 8 段
- 每段取一个代表点
- 选出最优 sector
- 第二轮在这个 sector 里再细分成 4 个 policy
- 总共只评 12 个点

### 图里应该有什么

- `Round 1 - Coarse (8 sectors)`
- 8 个小卡片 `S1 ... S8`
- 高亮一个，比如 `S3*`
- 向下箭头
- `Round 2 - Fine (4 policies in S3)`
- 4 个小卡片 `P1 ... P4`
- 高亮一个，比如 `P2*`
- 最后一个大红框 `Final policy selected`
- 小字 `Total evaluations: 8 + 4 = 12`

### 生成 prompt

```text
Create a clean poster-style panel titled "Coarse-to-Fine Policy Search". At the top, show a short sentence explaining that the frontier is divided into 8 evenly spaced sectors and one representative policy is evaluated in each sector. Then create a section called "Round 1 - Coarse (8 sectors)" with eight rounded boxes labeled S1 to S8 arranged horizontally. Highlight one winning sector, for example S3, using a crimson outline and stronger emphasis. Add a downward arrow with the caption "Select best sector, zoom in". Below that, create "Round 2 - Fine (4 policies in S3)" with four rounded boxes labeled P1 to P4, and highlight one winning policy, for example P2. At the bottom, show a strong crimson rounded rectangle saying "Final policy selected" and a smaller line "Total evaluations: 8 + 4 = 12". White background, flat vector academic poster style, very readable, similar to a polished conference poster panel.
```

## 5. Why Not Evolutionary Search 图

这张图非常值得加。  
它不是为了“黑进化算法”，而是为了说明：在你们这个 exact-budget 问题里，crossover 的 inheritance 会被 repair 破坏。

### 你要表达的是

- Parent A
- Parent B
- crossover child
- child 超预算
- repair step 重写大量 bit assignment
- 最终 child 不再保留有效继承结构

### 图里应该有什么

- 两条 parent policy 彩色条带
- 一个 crossover 箭头
- 生成 child
- child 上出现红色 warning：`over budget`
- 再经过 repair
- repair 之后的 child 颜色块大幅变化
- 最下方一句话：`repair destroys inherited structure`

### 生成 prompt

```text
Create an explanatory academic diagram titled "Why Not Evolutionary Search?" Show two parent mixed-precision policies as colored horizontal strips, Parent A and Parent B, composed of INT8 red segments, INT4 green segments, and INT2 gray segments. Use a crossover arrow to combine them into a child policy. Immediately show a red warning badge and a budget meter indicating that the child exceeds the exact budget. Then show a repair step that rewrites many segments of the child policy. The final repaired child should visibly differ from the original inherited structure. Add a clear callout saying "repair destroys inherited structure". White background, flat vector scientific poster style, crimson warning accents, clean arrows, minimal but strong explanatory labels.
```

## 6. 结果主图

这张图必须有。  
我建议做成双栏结果图：

- 左边：accuracy
- 右边：avg completion tokens

这样你就同时讲清楚：

- mixed 在哪些 benchmark 上比 INT4 更准
- mixed 在 token usage 上是不是更省

这会比单放 accuracy 更强。

### 你要表达的是

- MMLU-coding 上 mixed 和 INT4 持平
- HumanEval / BigCodeBench / MATH 上 mixed 高于 INT4
- token usage 上 mixed 在当前主线 benchmark 里都低于 INT4
- 尤其 BigCodeBench 这组最漂亮

### 图里应该有什么

- 左 panel：4 个 benchmark 分组柱状图，4 种颜色对应 `BF16 / INT8 / INT4 / Mixed`
- 右 panel：同样 4 个 benchmark 分组柱状图，纵轴换成 `avg completion tokens`
- Mixed 用红色或深红强调
- INT4 用绿色
- BF16 用深色
- INT8 用暖色
- 给 mixed vs INT4 画一个小箭头或高亮边框

### 生成 prompt

```text
Create a two-panel academic results figure for a conference poster. Left panel: grouped bar chart for benchmark accuracy comparing BF16, INT8, INT4, and Mixed across MMLU-coding, HumanEval, BigCodeBench-Hard, and MATH-500. Right panel: grouped bar chart for average completion tokens for the same benchmarks and model variants. Use consistent colors: BF16 dark navy, INT8 warm orange, INT4 muted green, Mixed crimson. Visually highlight that Mixed is higher than INT4 on HumanEval, BigCodeBench-Hard, and MATH-500, while tying INT4 on MMLU-coding. Also highlight that Mixed uses fewer completion tokens than INT4 across the current main benchmarks. Leave generous room for manual numeric labels and axis values. White background, flat vector scientific poster style, clean legend, subtle grid lines, publication quality.
```

## 7. 可选图：Workload-Family-Aware Policy 图

这张图不是必须，但如果你想把 story 讲得更严谨，我建议加。  
它能帮你避免别人误解成“每个 benchmark 都单独搜一个 policy”。

### 你要表达的是

- 不是 one policy per benchmark
- 是 one policy per workload family
- code benchmarks 共用一个 code policy
- math benchmark 用一个 math policy

### 图里应该有什么

- 左边一个 `Code policy` 卡片
- 下面连到 `MMLU-coding`, `HumanEval`, `BigCodeBench`
- 右边一个 `Math policy` 卡片
- 连到 `MATH-500`
- 每个 policy card 里可以写 bit counts，比如：
  - `code policy: 57 INT2 / 80 INT4 / 112 INT8`
  - `math policy: 75 INT2 / 64 INT4 / 110 INT8`

### 生成 prompt

```text
Create a clean academic infographic showing workload-family-aware mixed-precision policies. On the left, show a card labeled "Code-oriented policy" with a small bit-allocation strip and a note like "57 INT2 / 80 INT4 / 112 INT8". Draw arrows from this card to three benchmark icons labeled MMLU-coding, HumanEval, and BigCodeBench-Hard. On the right, show a card labeled "Math-oriented policy" with another bit-allocation strip and a note like "75 INT2 / 64 INT4 / 110 INT8". Draw an arrow from this card to MATH-500. Add a short caption such as "One policy per workload family, not one policy per benchmark". White background, flat vector MLSys poster style, crimson accents, very clean and interpretable.
```

## 如果版面有限，优先级这样排

如果你只能放 4 张图，我建议放：

- `Motivation: Uniform INT4 vs Mixed`
- `Method pipeline`
- `Coarse-to-fine search`
- `Results figure`

如果你能放 6 张，我建议就是：

- `Motivation 图`
- `Pipeline 图`
- `Policy frontier 图`
- `Coarse-to-fine 图`
- `Why not evolutionary 图`
- `Accuracy + token usage 结果图`
