# NER 微调：范围、模型弱点与数据决定

2026-09-19 定。目标：把 `xlm-roberta-base-ner-hrl` 全量微调成一个三语共用、可直接替换的模型
（标签集不变：`DATE / PER / ORG / LOC`，导出 ONNX + int8，`model.json` 不改结构）。不做按语言的 LoRA：
模型的问题是格式和文体（登记公告、邮件、表单），不是语言；`flutter_onnxruntime 1.8.5` 也没有运行时适配器接口。

数字来自当天 `tool/run_all_benchmarks.sh --no-stress` 的结果（`repairSpans` 之后）。

## 1. 进入微调范围的规则

判断标准：靠**格式或校验位**做决定的留在规则里；靠**上下文和语义**的交给模型。

| 组 | 规则 | 微调后 |
|---|---|---|
| 公司名 | `fr/es/gb/us:company`、`fr/es:company_head`、`fr:company_prefix`、`fr/es/gb:labeled_company`、`universal:institution_en`、`fr:court` | 法律后缀那几条降到模型之后（confidence < 0.8）留作安全网；其余消融后仍误遮的删除 |
| 街道 | `fr/es/gb/us:street`（以及 `be:street`、`ch:street-*`） | 降到模型之后 |
| 模糊的邮编 | `fr:postal`、`es:postal`、`us:zip_plus4`、`universal:postal_city` | 降到模型之后；`gb:postcode`、`ie:eircode` 不动 |
| 给模型打的补丁 | `titleStoplist`、`NerDetector.titleCased` | 变成训练样本；代码等消融结果再决定去留 |

不进：邮箱、IBAN、卡号、IP、MAC、URL、密钥、带校验位的证件号、各国电话、金额、日期（出生日期靠标签词判定）。
松散的数字规则（`long_number`、`labeled_id`、护照号、账号、`siret`）和参考号第一轮不做：需要新标签，检测必须是确定性的。

降级流程：训练数据覆盖该现象 → 关掉规则做消融（看泄露率和多遮）→ 模型召回不低于规则才降级 → 降级后还误遮的才删。
`repairSpans` 保证降级的规则在重叠里输掉也不会露字。

## 2. 模型目前的弱点（训练数据要定向覆盖）

1. **全大写姓名 / 公司，登记处语序"姓 姓 名"**：`PINCHAS ROZEN`、`CASTRO BALLESTEROS JUAN BAUTISTA`、`MOREL REGINE-JEANNE-CLAUDE`。
   公开文档集 PERSON 漏检 49、部分命中 38。
2. **小写、昵称、单独出现的名字**：`karen`、`tom`、`Ina`、`Kath`、`Mog`、`ben lee`。
3. **逗号格式的人名**：`Roche, Jean-François Jules`、`Hammond, Don`、`BOSQUET Cassandra, Stéphanie, Christiane`。
4. **没有法律后缀的商号、缩写、带撇号或数字的名字**：`LB INVEST`、`REST'OR`、`HK-RS`、`Net'Pro 43`、`EPMI`、`ENA`、
   `Brightwater Dental Practice`；以人名命名的事务所 `Russell McVeagh`、`Turpin Barker Armstrong`。
5. **类型混淆**：人名 → LOC（`SALA TORRES JOSE`、`Carmen`），公司 → PER（`BONNEVEINE MENAGER`），公司 → LOC（`JERVIS`）。
6. **跨行**："姓名\n部门"被合成一个 ORG：`Darnell Whitaker\nPatient Services`、`Priya Raman\nAccounts`（手机实测）。
7. **官方机构、公报名、登记处缩写当成实体**（多遮的最大来源，公开文档集 ORG 误报 47）：
   `BOLETÍN OFICIAL DEL REGISTRO MERCANTIL` ×11、`BODACC`、`R.M. MAHON`、`CNAE`、`Sociedad`、`Administración`。应为 `O`。
8. **完整邮政地址不是一个实体**：模型只认城镇名，街道靠规则。公开文档集 ADDRESS 漏检 48、部分命中 38；
   单独出现的城镇有时也漏（`Arras`、`Boulogne-sur-Mer`、`Paris`、`Bristol`）。
9. **职务词、部门词当成 PER / ORG**：`Chair`、`CFO`、`Head of Operations`。应为 `O`。
10. **正常大小写的西语双姓全名**：`Rocío Montenegro Díaz`、`Andrés Quintana Marín`。

## 3. 数据决定

- **生成方式**：模板为主 + 少量逐篇。子 agent 写带槽位的文档模板和实体清单（人名、商号、地址写法），脚本批量填充并做增强
  （全大写、全小写、"姓, 名"、"姓 姓 名"、名字后换行、OCR 噪声）；另由子 agent 逐篇写几百份完整文档保证自然度。
- **真实数据**：加入。BODACC、BORME、Companies House 开放数据的结构化字段（公司名、地址、人名）与原文做字符串对齐得到标注。
  取与测试集不同日期、不同公司的公告。
- **语言**：只做英语、法语、西班牙语。中文及底座的其他语言如果退化，接受；手写基准里的中文用例只用来记录退化幅度。
- **带地名的地方机构**（`CPAM de Nantes`、`Tribunal de Commerce de Reims`、`Juzgado de lo Mercantil de Sevilla`）：
  机构名不标，**其中的地名标成 ADDRESS**。与 `fr:court` 规则的现有行为一致。
  已同步（2026-09-19）：`benchmark/public/ANNOTATION.md` 和 9 条受影响的标签已改，基线已重跑（见交接文档第 2 节）。

## 4. 默认做法

- 标注口径沿用 `benchmark/public/ANNOTATION.md`（除上面那条）：同一行内连续的地址是一个实体，跨行按行拆；头衔、职务不标；
  法院和政府机构不标；`the Company`、`la société` 这类泛称不标。训练标签与测试标签必须同口径。
- 日期照常标 `DATE`，否则模型会忘掉这个标签。不新增标签。
- 标注用行内标记（如 `[[PERSON|Jean Dupont]]`），由脚本转成字符偏移，不让生成方自己数偏移。
- **训练集与测试集隔离**：`benchmark/public_cases.json` 的 45 份文档、`benchmark/synthetic_cases.json` 及
  `tool/generate_synthetic_cases.py` 的模板和人名表、`benchmark/ner_cases.json`、`benchmark/hard_negatives.json` 一律不进训练。
- 生成训练数据之前，先由另一组子 agent 写一套**消费者文档测试集**（信件、发票、工资单、病历、租约；英法西）并冻结：
  用户粘贴的是这类文本，不是登记公告。
- 学会小写名字之后 `will`、`rose`、`mark`、`price` 容易被误遮：训练数据里放这类普通词的负样本，`hard_negatives.json` 同步加句子。
- 生成器和模板进仓库，生成出的大文件 git-ignore。
- 实现见 `training/README.md`（2026-09-20）：清单 / 模板 / 逐篇文档由隔离的子 agent 写，`build.py` 填充并增强，
  `fetch_registry.py` 从 BODACC、BORME、The Gazette 的结构化字段对齐出银标，`check_isolation.py` 守住与四套测试集的隔离。
- 上线前核查所用语料的许可；用到公开 NER 语料（CoNLL-2003、WikiNEuRal、MultiNERD）前先确认可商用，默认不用。
- 量化后重跑全部基准：int8 有时会抹掉一部分微调收益。
