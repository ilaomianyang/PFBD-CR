import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const finalDir = path.resolve(
  "experiments/trustmeteo_qc/submission_final_20261009",
);
const databasePath = path.join(finalDir, "final_experiment_evidence.sqlite");
const outputPath = path.join(finalDir, "final_experiment_evidence.xlsx");
const db = new DatabaseSync(databasePath, { readOnly: true });
const workbook = Workbook.create();
const font = "Arial";

const metricNotes = {
  precision: "精确率 = TP / (TP + FP)。表示已发出告警中命中攻击的比例；仅对应本表记录的阈值点，不是与阈值无关的指标。",
  recall: "召回率 = TP / (TP + FN)。事件级表现还要看 event_coverage，因为逐行召回率可能掩盖整段事件漏检。",
  f1: "F1 = 2 * 精确率 * 召回率 / (精确率 + 召回率)。本行的汇总口径决定样本权重；不同汇总口径不能直接当作同一个估计值比较。",
  false_alarm_rate: "误报率 = FP / (FP + TN)，分母按来源表定义。它不是“所有告警中误报所占比例”。",
  event_coverage: "事件覆盖率 = 被检出的攻击事件数 / 攻击事件总数。事件识别和汇总规则以来源表为准；它不同于逐行召回率。",
  average_precision: "平均精确率（AP），来自保存的逐次运行分数/排序证据。仅凭 TP、FP、FN、TN 不能重新计算。",
  roc_auc: "ROC 曲线下面积，来自保存的逐次运行分数/排序证据。仅凭 TP、FP、FN、TN 不能重新计算。",
  count_pooled: "按计数合并：先合并底层混淆矩阵计数，再计算指标。因此数据量较大的提供方或站点权重更高。",
  provider_seed_macro: "提供方-随机种子宏平均：对 20 个“提供方×随机种子”指标取算术平均。它不是提供方总体估计，也不是按样本数合并的结果。",
  provider_equal: "提供方等权：先分别按每个提供方的计数计算指标，再让 4 个提供方权重相同。",
  station_equal: "站点等权：先按站点汇总随机种子，再让 42 个站点权重相同。它表示本数据内的站点平衡，不代表对更大站点总体的独立验证。",
  ci95_low: "按相邻列所述抽样方法计算的区间下限。解释时必须保留抽样/聚类单位；不能自动视为总体置信区间。",
  ci95_high: "按相邻列所述抽样方法计算的区间上限。解释时必须保留抽样/聚类单位；不能自动视为总体置信区间。",
  bh_q: "按 Benjamini-Hochberg 方法校正后的 q 值。报告显著性时必须同时保留检验对象和校正检验族。",
  test: "来源文件记录的检验或比较名称。解释前请核对相邻的检验族、估计口径和多重比较校正信息。",
  exploratory: "探索性标记。TRUE/1（真/1）表示探索或冻结后敏感性证据，包括阶段500；不得替代 V6 主结果。",
  tp: "真正例：按来源表的事件/行匹配规则记录或重建。",
  fp: "假正例：按来源表的事件/行匹配规则记录或重建。",
  fn: "假负例：按来源表的事件/行匹配规则记录或重建。",
  tn: "真负例：按来源表的事件/行匹配规则记录或重建。",
};

const tableNotes = {
  csv_final_v6_method_summary: "V6 主证据，按“提供方×随机种子”汇总。包含 20 个运行均值，不是按计数合并的结果。",
  csv_final_v6_aggregation_levels_corrected: "V6 主证据，比较不同统计口径。按计数合并、提供方等权、提供方-随机种子宏平均、站点等权必须分别解释。",
  csv_final_v6_scenario_ci_corrected: "V6 主证据，按场景计算不确定性。保留所有行以及抽样方法和估计口径字段。",
  csv_final_v6_station_cluster_ci_corrected: "V6 主证据，站点聚类不确定性分析。抽样单位是站点；不代表已经证明可推广到更大站点总体。",
  csv_final_v6_union_decision_provenance: "V6 主证据，记录 trust_shape_union 的决策来源。用于核查联合判定和告警来源。",
  csv_final_v6_scenario_wins: "V6 主证据，按场景与基线比较。支持有限范围内的多数优势或混合结果，不支持全面领先。",
  csv_final_external_transfer: "外部迁移证据。保留负面和混合结果，不要称为广泛的外部验证。",
  csv_final_negative_boundaries: "负面结果和限制，用于界定当前证据能支持的结论范围。",
  csv_final_method_implementation_boundary_corrected: "方法实现一致性与比较边界。代理基线不能描述成对指定方法的精确复现。",
  csv_final_baseline_resources_corrected: "修正后的基线资源统计。仅在文件说明的测量范围内比较。",
  csv_final_trust_resources_corrected: "修正后的本文方法资源统计。除非测量范围覆盖整套系统，否则不能据此推断生产总成本。",
  csv_final_testbed_resources_corrected: "经授权的本地回环测试台资源数据，与 V6 批处理性能分开解释。",
  csv_final_online_detection: "在线检测/测试台证据。不要与使用居中窗口的 V6 批处理评估混为一谈。",
  csv_final_online_offline_recovery: "在线与离线恢复比较。离线双向插值不等同于因果在线恢复。",
  csv_final_operating_points: "运行阈值敏感性。将阈值视为可部署阈值前，先核对阈值选择过程。",
  csv_stage500_final_candidate_runs: "仅作探索性分析。阶段500是冻结后的敏感性证据，不能替代 V6 主结果。",
  csv_stage500_final_candidate_scenario: "仅作探索性分析。阶段500场景结果不是确认性证据，也不能与 V6 合并。",
};

const sheetNameMap = {
  README: "使用说明",
  Paper_Source_Map: "论文引用索引",
  Artifact_Catalog: "文件清单",
  Primary_Metrics: "核心结果",
  Workbook_Index: "工作表索引",
  JSON_Audits: "JSON审计原文",
  Audit_Reports: "审计报告原文",
  Manuscript_Materials: "论文材料原文",
  Data_Dictionary: "字段中文释义",
};

const wordZh = {
  alert: "告警", alerts: "告警", alarm: "告警", count: "数量", mean: "均值",
  average: "平均", precision: "精确率", recall: "召回率", false: "误报",
  rate: "率", event: "事件", coverage: "覆盖率", attack: "攻击", row: "行",
  rows: "行数", normal: "正常", detected: "检出", events: "事件数",
  seed: "随机种子", provider: "提供方", station: "站点", scenario: "场景",
  method: "方法", baseline: "基线", candidate: "候选方法", selected: "选定",
  valid: "验证集", test: "测试集", train: "训练", run: "运行", runs: "运行次数",
  f1: "F1分数", roc: "ROC", auc: "曲线下面积", averageprecision: "平均精确率",
  average_precision: "平均精确率", ap: "AP", far: "误报率", bh: "BH校正",
  q: "q值", ci95: "95%区间", low: "下限", high: "上限", lower: "下限",
  upper: "上限", tp: "真正例", fp: "假正例", fn: "假负例", tn: "真负例",
  iteration: "迭代", iterations: "迭代次数", bootstrap: "自助抽样",
  resampling: "重抽样", unit: "单位", type: "类型", family: "类别",
  detail: "明细", summary: "汇总", aggregate: "聚合", aggregation: "汇总口径",
  corrected: "修正版", correction: "校正", boundary: "边界", resource: "资源",
  latency: "延迟", seconds: "秒", ms: "毫秒", mb: "兆字节", parameter: "参数",
  parameters: "参数", threshold: "阈值", score: "分数", mean: "均值", std: "标准差",
  probability: "概率", pass: "通过", passes: "通过数", win: "胜", wins: "胜场数",
  loss: "负", losses: "负场数", tie: "平", ties: "平局数", share: "占比",
  fraction: "比例", total: "总计", target: "目标", constraint: "约束",
  constrained: "受约束", gate: "门槛", extended: "扩展", formal: "正式",
  implementation: "实现", status: "状态", source: "来源", evidence: "证据",
  layer: "层级", online: "在线", offline: "离线", causal: "因果",
  recovery: "恢复", delay: "延迟", drop: "下降", replay: "重放",
  structured: "结构化", subtle: "细微", coordinated: "协同", offset: "偏移",
  drift: "漂移", noise: "噪声", spike: "尖峰", short: "短", long: "长",
  stuck: "卡滞", valid: "验证集", validation: "验证", testbed: "测试台",
  external: "外部", transfer: "迁移", provider: "提供方", station: "站点",
  selected: "选定", selection: "选择", rule: "规则", weight: "权重",
  weights: "权重", difference: "差值", delta: "变化量", gap: "差距",
  margin: "裕量", budget: "预算", ratio: "比值", mean: "均值",
  minimum: "最小", maximum: "最大", min: "最小", max: "最大",
  name: "名称", id: "编号", label: "标签", kind: "类别", notes: "备注",
  note: "备注", description: "说明", rows_scored: "评分行数", frames: "帧数",
  throughput: "吞吐量", fit: "拟合", seconds: "秒", observed: "观测",
  deployed: "部署", deployable: "可部署", trust: "可信", union: "联合",
  fusion: "融合", hybrid: "混合", stage: "阶段", exact: "精确",
  exact_agree: "完全一致", family_count: "类别数", count_pooled: "按计数合并",
  provider_equal_counts: "提供方等权", provider_seed_macro: "提供方-随机种子宏平均",
  station_equal_counts: "站点等权",
  authoritative: "权威", tables: "数据表", table: "数据表", claim: "论文主张",
  area: "主题", writing: "写作", artifact: "文件", included: "已纳入",
  database: "数据库", column: "字段", columns: "字段数", path: "路径",
  sha256: "SHA256校验值", class: "类别", rule: "规则", source: "来源",
  index: "索引", text: "文本", markdown: "Markdown", json: "JSON",
  file: "文件", report: "报告", manuscript: "论文", material: "材料",
  iteration: "迭代", fraction: "比例", share: "占比", estimate: "估计值",
  estimand: "统计口径", pooled: "合并", macro: "宏平均", equal: "等权",
  constraint: "约束", outcome: "结果", test: "测试", selected: "选定",
  exact: "精确", corrected: "修正", resource: "资源", online: "在线",
  offline: "离线", recovery: "恢复", runtime: "运行时间", heterogeneity: "异质性",
  transfer: "迁移", negative: "负面", boundary: "边界", comparison: "比较",
  providers: "提供方数量", stations: "站点数量", false_alarm_rate: "误报率",
  event_coverage: "事件覆盖率", attack_rows: "攻击样本行数",
  normal_rows: "正常样本行数", detected_events: "检出事件数",
  average_precision: "平均精确率", roc_auc: "ROC曲线下面积",
  ci95_low: "95%区间下限", ci95_high: "95%区间上限",
};

const valueZh = {
  pooled_counts: "按计数合并",
  count_pooled: "按计数合并",
  provider_equal_counts: "提供方等权",
  provider_equal_count_metrics: "提供方等权指标",
  provider_seed_macro: "提供方-随机种子宏平均",
  macro_provider_seed: "提供方-随机种子宏平均",
  station_equal_counts: "站点等权",
  station_equal_count_metrics: "站点等权指标",
  "V6 frozen primary": "V6 冻结主证据",
  "primary plus exploratory": "主证据及探索性证据",
  exploratory: "探索性证据",
  "external and limitation evidence": "外部迁移与限制证据",
  "authorized testbed and safety analysis": "授权测试台与安全性分析",
};
const sourceMapZh = {
  primary_metrics: "主要指标",
  scenario_and_uncertainty: "场景结果与不确定性",
  baseline_and_implementation: "基线方法与实现边界",
  external_and_negative_boundaries: "外部迁移与负面结果边界",
  recovery_and_runtime: "恢复能力与运行资源",
  exploratory_closeout: "探索性结果收口",
  "Use corrected count-pooled and explicitly labeled macro/weighted rows; do not replace with Stage500 exploratory values.":
    "使用修正后的按计数合并结果，并明确标记宏平均/加权口径；不得用阶段500探索性结果替换。",
  "Report event coverage, F1, FAR, event counts, and dependency-aware intervals together.":
    "同时报告事件覆盖率、F1、误报率、事件数量及考虑数据依赖关系的区间。",
  "Name proxy implementations accurately and keep formal MLP and other post-freeze comparisons exploratory.":
    "准确称呼代理实现；正式 MLP 和其他冻结后比较均保持为探索性结果。",
  "Keep FMI/SMHI mixed transfer, OCSVM complementarity, and failed scenarios visible.":
    "保留 FMI/SMHI 混合迁移结果、OCSVM 的互补性及失败场景。",
  "Do not equate batch V6 detection or offline recovery with strict online causal capability.":
    "不得将 V6 批量检测或离线恢复等同于严格的在线因果能力。",
  "Use only as sensitivity/ablation context; never substitute for the V6 primary result.":
    "仅用于敏感性/消融背景，绝不能替代 V6 主结果。",
};
const fieldNotes = {
  precision: metricNotes.precision,
  recall: metricNotes.recall,
  f1: metricNotes.f1,
  false_alarm_rate: metricNotes.false_alarm_rate,
  event_coverage: metricNotes.event_coverage,
  average_precision: metricNotes.average_precision,
  roc_auc: metricNotes.roc_auc,
  tp: metricNotes.tp,
  fp: metricNotes.fp,
  fn: metricNotes.fn,
  tn: metricNotes.tn,
  estimand: "统计口径名称。用于区分按计数合并、提供方等权、宏平均和站点等权等不同估计目标。",
  aggregation: "聚合/加权方式。须与 estimand 一起阅读。",
  ci95_low: metricNotes.ci95_low,
  ci95_high: metricNotes.ci95_high,
  resampling_unit: "计算区间时进行重抽样的单位。置信区间解释不得脱离此单位。",
  resampling: "区间估计使用的重抽样方法。",
  bh_q: metricNotes.bh_q,
  exploratory: metricNotes.exploratory,
  provider: "天气数据提供方标识。V6 主评估覆盖哪些提供方，以论文方法和证据层级表为准。",
  station_id: "气象站点标识。站点数不等同于独立外部站点总体样本量。",
  seed: "随机种子编号。不同种子用于重复运行，不代表独立提供方。",
  scenario: "模拟攻击/测试场景名称。需要与攻击参数及失败边界表对照解释。",
  method: "方法或基线标识。技术名称按原始记录保留。",
  notes: "来源材料附带的说明。除明确修正表外，内容按原文保留。",
};
const primaryNoteZh = {
  "TP/FP/FN/TN reconstructed from saved metric definitions; event coverage is event-count weighted. AP/ROC are not recomputable from counts.":
    "TP、FP、FN、TN 按保存的指标定义重建；事件覆盖率按事件数量加权。仅凭这些计数无法重算平均精确率和 ROC 曲线下面积。",
  "Each provider is first summarized from its own counts; providers are then equally weighted. AP/ROC are not recomputable from counts.":
    "先按每个提供方自己的计数计算指标，再对提供方等权平均。仅凭这些计数无法重算平均精确率和 ROC 曲线下面积。",
  "Mean of the 20 provider-seed metrics; retained for comparison with the frozen Stage349 summary. It is not a count-pooled estimate and is not a provider-population estimate.":
    "对 20 个“提供方×随机种子”指标取平均，用于与冻结的阶段349汇总对照。它不是按计数合并的估计，也不代表提供方总体估计。",
  "Each station is first summarized across seeds from its own counts; stations are then equally weighted. AP/ROC are not available in the frozen station table.":
    "先对每个站点的各随机种子计数进行汇总，再对站点等权平均。冻结的站点表没有平均精确率和 ROC 曲线下面积。",
};

function safeSheetName(name, used) {
  let out = name.replace(/[\\/?*:[\]]/g, "_").slice(0, 31);
  if (!out) out = "数据";
  const base = out;
  let i = 2;
  while (used.has(out.toLowerCase())) {
    const suffix = `_${i++}`;
    out = `${base.slice(0, 31 - suffix.length)}${suffix}`;
  }
  used.add(out.toLowerCase());
  return out;
}

function translateField(name) {
  const key = String(name).toLowerCase();
  if (wordZh[key]) return wordZh[key];
  const parts = key.split("_");
  return parts.map((part) => wordZh[part] || part.toUpperCase()).join(" ");
}

function bilingualHeader(name) {
  const zh = translateField(name);
  return `${name}（${zh}）`;
}

function displayTableName(name) {
  const table = name.replace(/^csv_/, "");
  if (table === "final_v6_aggregation_levels_corrected") return "V6主结果_统计口径修正";
  if (table === "final_v6_method_summary") return "V6方法指标汇总";
  if (table === "final_v6_scenario_wins") return "V6场景胜负统计";
  if (table === "final_v6_scenario_ci_corrected") return "V6场景置信区间修正";
  if (table === "final_v6_station_cluster_ci_corrected") return "V6站点聚类置信区间";
  if (table === "final_v6_union_decision_provenance") return "V6联合判定来源";
  if (table.startsWith("stage")) {
    const match = table.match(/^stage(\d+)_(.*)$/);
    if (match) return `探索${match[1]}_${translateField(match[2])}`;
  }
  return `数据_${translateField(table)}`;
}

function sheetDescription(tableName) {
  if (tableNotes[tableName]) return tableNotes[tableName];
  if (tableName.startsWith("csv_stage500_")) {
    return "仅作探索性分析。阶段500属于冻结后的敏感性证据，不能替代 V6 主结果，也不能写成确认性结论。";
  }
  if (tableName.startsWith("csv_stage")) {
    return "冻结后的探索性/敏感性分析表。保留用于追溯，请与 V6 主证据区分。";
  }
  if (tableName.startsWith("csv_final_v6_")) {
    return "V6 证据表。请结合“论文引用索引”及本表统计口径、来源字段判断其属于主证据还是补充证据。";
  }
  if (tableName.startsWith("csv_final_")) {
    return "最终材料中的补充证据。请结合范围、来源和限制字段解释，不要超出证据支持范围。";
  }
  return "此表从权威证据数据库复制，原始记录和值均予以保留。";
}

function asExcelValue(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "bigint") return Number(value);
  if (Buffer.isBuffer(value)) return value.toString("base64");
  return value;
}

function columnLetter(n) {
  let s = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function addAnnotation(sheet, address, text) {
  const note = text.length > 30000 ? `${text.slice(0, 29950)} [truncated in Excel note]` : text;
  workbook.notes.add({
    id: `${sheet.name}:${address}`,
    target: { cell: { sheetName: sheet.name, sheetId: sheet.sheetId, address } },
    authorId: "",
    createdAt: "",
    body: { plainText: note },
  });
}

function writeRows(sheet, headers, rows, maxRows = 5000) {
  const width = Math.max(headers.length, 1);
  sheet.getRangeByIndexes(0, 0, 1, width).values = [
    headers.map((v) => String(v)),
  ];
  sheet.getRange("A1").format = {
    fill: "#24445C",
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  if (headers.length > 1) {
    sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
      fill: "#24445C",
      font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
      wrapText: true,
      horizontalAlignment: "center",
      verticalAlignment: "center",
    };
  }
  sheet.getRange("A1").format.rowHeight = 30;

  for (let start = 0; start < rows.length; start += maxRows) {
    const chunk = rows.slice(start, start + maxRows);
    const values = chunk.map((row) => row.map(asExcelValue));
    for (const [rowIndex, valuesRow] of values.entries()) {
      for (const [colIndex, value] of valuesRow.entries()) {
        if (typeof value === "string" && value.length > 32767) {
          throw new Error(`Excel text limit exceeded: ${sheet.name}!${columnLetter(colIndex + 1)}${start + rowIndex + 2}`);
        }
      }
    }
    sheet.getRangeByIndexes(start + 1, 0, values.length, width).values = values;
  }

  const lastRow = rows.length + 1;
  if (lastRow > 1) {
    sheet.getRangeByIndexes(1, 0, lastRow - 1, width).format = {
      font: { name: font, size: 10, color: "#202B33" },
      verticalAlignment: "center",
    };
    sheet.getRangeByIndexes(0, 0, lastRow, width).format.borders = {
      insideHorizontal: { style: "thin", color: "#E5EAEE" },
    };
  }
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  return lastRow;
}

function formatByHeader(sheet, headers, rowCount) {
  const totalRows = Math.max(rowCount - 1, 1);
  headers.forEach((raw, index) => {
    const key = String(raw).toLowerCase();
    const col = columnLetter(index + 1);
    const dataRange = sheet.getRange(`${col}2:${col}${rowCount}`);
    if (/^(f1|precision|recall|false_alarm_rate|event_coverage|average_precision|roc_auc|bh_q|q_value|p_value|pvalue|ci95_low|ci95_high|lower|upper|mean|std|se|score|auc|ap)$/.test(key)) {
      dataRange.format.numberFormat = "0.0000";
    } else if (/^(tp|fp|fn|tn|events|detected_events|attack_rows|normal_rows|providers|stations|runs|seed|n|count|row_count|column_count)$/.test(key)) {
      dataRange.format.numberFormat = "#,##0";
    }
    const noteText = metricNotes[key];
    if (noteText) addAnnotation(sheet, `${col}1`, noteText);
  });
  if (headers.length) {
    sheet.getRangeByIndexes(0, 0, rowCount, headers.length).format.wrapText = false;
    sheet.getRangeByIndexes(0, 0, 1, headers.length).format.wrapText = true;
    for (let i = 0; i < headers.length; i++) {
      const col = columnLetter(i + 1);
      sheet.getRange(`${col}1:${col}${rowCount}`).format.columnWidth = Math.min(
        42,
        Math.max(18, bilingualHeader(headers[i]).length + 2),
      );
    }
    sheet.getRange(`A1:${columnLetter(headers.length)}1`).format.rowHeight = 30;
  }
}

function readTable(name) {
  const quoted = `"${name.replaceAll('"', '""')}"`;
  const columns = db.prepare(`PRAGMA table_info(${quoted})`).all().map((c) => c.name);
  const rows = db.prepare(`SELECT * FROM ${quoted}`).all().map((row) => columns.map((c) => row[c]));
  return { columns, rows };
}

function writeDataSheet(name, workbookTable, usedNames) {
  const { columns, rows } = readTable(name);
  const sheet = workbook.worksheets.add(safeSheetName(workbookTable, usedNames));
  let visibleRows = rows;
  if (name === "paper_source_map") {
    visibleRows = rows.map((row) =>
      row.map((value) => sourceMapZh[value] || valueZh[value] || value),
    );
  } else if (name === "artifact_catalog") {
    const notesIndex = columns.indexOf("notes");
    visibleRows = rows.map((row) => {
      const out = [...row];
      if (notesIndex >= 0 && typeof out[notesIndex] === "string") {
        out[notesIndex] = out[notesIndex]
          .replace("query table:", "数据库查询表：")
          .replace("raw JSON retained in json_artifacts", "原始 JSON 保存在“JSON审计原文”工作表")
          .replace("writing/audit report retained in report_text", "写作/审计报告原文保存在“审计报告原文”工作表")
          .replace("kept as external binary artifact; source path and hash retained", "未展开的二进制文件；已保留原路径和 SHA256 校验值");
      }
      return out;
    });
  }
  const lastRow = writeRows(sheet, columns.map(bilingualHeader), visibleRows, 5000);
  formatByHeader(sheet, columns, lastRow);
  addAnnotation(sheet, "A1", sheetDescription(name));
  return { tableName: name, sheetName: sheet.name, rows: rows.length, columns: columns.length };
}

function splitOriginalText(text) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + 30000, text.length);
    // Excel counts UTF-16 units. Keep surrogate pairs and CRLF together.
    if (end < text.length && /[\uD800-\uDBFF]/.test(text[end - 1])) end -= 1;
    if (end < text.length && text[end - 1] === "\r" && text[end] === "\n") end -= 1;
    let lineFeeds = 0;
    for (let i = start; i < end; i++) {
      if (text[i] === "\n" && ++lineFeeds === 250) {
        end = i + 1;
        break;
      }
    }
    chunks.push(text.slice(start, end));
    start = end;
  }
  if (!chunks.length) chunks.push("");
  if (chunks.join("") !== text) throw new Error("Original text segmentation lost content");
  return chunks;
}

function writeTextSheet(tableName, sheetTitle, usedNames) {
  const { rows } = readTable(tableName);
  return writeOriginalTextRows(rows, sheetTitle, usedNames);
}

function writeOriginalTextRows(rows, sheetTitle, usedNames) {
  const sheet = workbook.worksheets.add(safeSheetName(sheetTitle, usedNames));
  const segmentedRows = rows.flatMap(([name, source, hash, content]) => {
    const chunks = splitOriginalText(content);
    return chunks.map((chunk, index) => [name, source, hash, chunk, index + 1, chunks.length]);
  });
  const header = ["文件名", "原始路径", "SHA256校验值", "原文内容（分段）", "段号", "总段数"];
  const lastRow = writeRows(sheet, header, segmentedRows, 5000);
  formatByHeader(sheet, header, lastRow);
  sheet.getRange(`A1:A${lastRow}`).format.columnWidth = 36;
  sheet.getRange(`B1:B${lastRow}`).format.columnWidth = 64;
  sheet.getRange(`C1:C${lastRow}`).format.columnWidth = 66;
  sheet.getRange(`D1:D${lastRow}`).format.columnWidth = 100;
  sheet.getRange(`D2:D${lastRow}`).format.wrapText = true;
  sheet.getRange(`D2:D${lastRow}`).format.verticalAlignment = "top";
  sheet.getRange(`A2:F${lastRow}`).format.rowHeight = 60;
  sheet.getRange(`E1:F${lastRow}`).format.columnWidth = 10;
  sheet.getRange(`E2:F${lastRow}`).format.numberFormat = "0";
  addAnnotation(sheet, "D1", "长原文按每段不超过30,000个UTF-16字符、250个换行分段，避免超过Excel单元格限制。同一文件按段号升序连接D列（不要额外加入换行或分隔符），即可还原完整原文。SHA256为来源文件校验值，不是单段校验值。");
  addAnnotation(sheet, "A1", "此页保留原始文件文字。原文不作改写；长原文分段保存，段号及总段数用于完整还原。阅读实验结果请优先查看结构化数据表和中文说明。");
  return { sheetName: sheet.name, rows: segmentedRows.length, sourceFiles: rows.length, columns: header.length };
}

function addPaperReadySheet(usedNames) {
  const sheet = workbook.worksheets.add(safeSheetName("论文直用数据", usedNames));
  sheet.showGridLines = false;
  let row = 1;
  let dataRowCount = 0;
  const formulaRows = [];

  function getRows(name) {
    const data = readTable(name);
    return {
      rows: data.rows,
      index: Object.fromEntries(data.columns.map((column, i) => [column, i])),
    };
  }

  function section(title, headers, rows, widths, formats = {}) {
    sheet.getRange(`A${row}`).values = [[title]];
    sheet.getRange(`A${row}:${columnLetter(headers.length)}${row}`).format = {
      fill: "#DCE6ED",
      font: { name: font, size: 11, bold: true, color: "#1E3545" },
      verticalAlignment: "center",
    };
    row += 1;
    sheet.getRangeByIndexes(row - 1, 0, 1, headers.length).values = [headers];
    sheet.getRangeByIndexes(row - 1, 0, 1, headers.length).format = {
      fill: "#24445C",
      font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
      wrapText: true,
      horizontalAlignment: "center",
      verticalAlignment: "center",
    };
    sheet.getRange(`A${row}:${columnLetter(headers.length)}${row}`).format.rowHeight = 32;
    row += 1;
    if (rows.length) {
      sheet.getRangeByIndexes(row - 1, 0, rows.length, headers.length).values = rows;
      sheet.getRangeByIndexes(row - 1, 0, rows.length, headers.length).format = {
        font: { name: font, size: 10, color: "#202B33" },
        verticalAlignment: "center",
        wrapText: false,
      };
      for (const [col, format] of Object.entries(formats)) {
        sheet.getRange(`${col}${row}:${col}${row + rows.length - 1}`).format.numberFormat = format;
      }
      dataRowCount += rows.length;
      row += rows.length;
    }
    row += 2;
  }

  const methods = getRows("csv_final_v6_method_summary");
  const methodRows = methods.rows
    .slice()
    .sort((a, b) => String(a[methods.index.method]).localeCompare(String(b[methods.index.method])))
    .map((r) => [
      r[methods.index.method], r[methods.index.precision], r[methods.index.row_recall],
      r[methods.index.f1], r[methods.index.event_coverage], r[methods.index.average_precision],
      r[methods.index.false_alarm_rate], r[methods.index.roc_auc],
    ]);
  formulaRows.push([
    "一、V6方法总体比较",
    "全部方法，逐方法行",
    "final_v6_method_summary.csv；各方法；四提供方×五随机种子，共20个提供方-种子运行。",
    "先在每个提供方-种子运行内依定义计算指标，再对20个运行的指标取算术平均。精确率=TP/(TP+FP)；逐行召回率=TP/(TP+FN)；F1=2TP/(2TP+FP+FN)；误报率=FP/(FP+TN)；事件覆盖率=检出攻击事件数/攻击事件总数。AP和ROC-AUC来自分数排序证据，不能仅由TP、FP、FN、TN重算。",
  ]);
  section(
    "一、V6方法总体比较（论文主表/图数据）",
    ["方法", "精确率", "逐行召回率", "F1", "事件覆盖率", "AP", "误报率", "ROC-AUC"],
    methodRows, [30, 12, 14, 12, 14, 12, 12, 12],
    { B: "0.0000", C: "0.0000", D: "0.0000", E: "0.0000", F: "0.0000", G: "0.0000", H: "0.0000" },
  );

  const aggs = getRows("csv_final_v6_aggregation_levels_corrected");
  const aggRows = aggs.rows.filter((r) => r[aggs.index.method] === "trust_shape_union").map((r) => [
    valueZh[r[aggs.index.estimand]] || r[aggs.index.estimand],
    valueZh[r[aggs.index.aggregation]] || r[aggs.index.aggregation],
    r[aggs.index.providers], r[aggs.index.stations],
    r[aggs.index.runs], r[aggs.index.tp], r[aggs.index.fp], r[aggs.index.fn],
    r[aggs.index.tn], r[aggs.index.attack_rows], r[aggs.index.normal_rows],
    r[aggs.index.events], r[aggs.index.detected_events], r[aggs.index.precision],
    r[aggs.index.recall], r[aggs.index.f1], r[aggs.index.false_alarm_rate],
    r[aggs.index.event_coverage], r[aggs.index.average_precision], r[aggs.index.roc_auc],
  ]);
  formulaRows.push([
    "二、Trust主结果的不同统计口径",
    "method=trust_shape_union；四个估计目标各一行",
    "final_v6_aggregation_levels_corrected.csv；V6四提供方、五随机种子。统计目标必须按“估计目标”列区分。",
    "按计数合并：先求和TP/FP/FN/TN，再计算Precision=TP/(TP+FP)、Recall=TP/(TP+FN)、F1=2TP/(2TP+FP+FN)、FAR=FP/(FP+TN)。提供方等权：各提供方先按各自计数计算，再对4个提供方等权平均。提供方-种子宏平均：对20个提供方×种子指标取算术平均。站点等权：每站点跨种子合并计数后计算指标，再对42站点等权平均。事件覆盖率=检出事件数/事件总数。AP、ROC-AUC无法从计数重算，缺失值须保留为空。",
  ]);
  section(
    "二、Trust主结果的不同统计口径（不可混用）",
    ["估计目标", "聚合方式", "提供方数", "站点数", "运行/聚合数", "TP", "FP", "FN", "TN", "攻击行数", "正常行数", "事件总数", "检出事件数", "精确率", "召回率", "F1", "误报率", "事件覆盖率", "AP", "ROC-AUC"],
    aggRows, [28, 28, 12, 12, 14, 12, 12, 12, 14, 14, 14, 12, 14, 12, 12, 12, 12, 14, 12, 12],
    { C: "#,##0", D: "#,##0", E: "#,##0", F: "#,##0", G: "#,##0", H: "#,##0", I: "#,##0", J: "#,##0", K: "#,##0", L: "#,##0", M: "#,##0", N: "0.0000", O: "0.0000", P: "0.0000", Q: "0.0000", R: "0.0000", S: "0.0000", T: "0.0000" },
  );

  const scenarios = getRows("csv_final_v6_scenario_aggregate");
  const scenarioRows = scenarios.rows
    .filter((r) => r[scenarios.index.method] === "trust_shape_union")
    .sort((a, b) => String(a[scenarios.index.scenario]).localeCompare(String(b[scenarios.index.scenario]))
    ).map((r) => [
      r[scenarios.index.scenario], r[scenarios.index.runs], r[scenarios.index.events],
      r[scenarios.index.detected_events], r[scenarios.index.tp], r[scenarios.index.fp],
      r[scenarios.index.fn], r[scenarios.index.tn], r[scenarios.index.precision],
      r[scenarios.index.row_recall], r[scenarios.index.f1],
      r[scenarios.index.false_alarm_rate], r[scenarios.index.event_coverage],
      r[scenarios.index.average_precision], r[scenarios.index.roc_auc],
    ]);
  formulaRows.push([
    "三、按攻击场景的最终表现",
    "method=trust_shape_union；9个攻击场景各一行",
    "final_v6_scenario_aggregate.csv；method=trust_shape_union；每场景四提供方×五种子。保留TP/FP/FN/TN用于手动复算。各场景正负样本范围按冻结协议，其他攻击场景不计入该行误报分母。",
    "精确率=TP/(TP+FP)；逐行召回率=TP/(TP+FN)；F1=2TP/(2TP+FP+FN)；误报率=FP/(FP+TN)；事件覆盖率=该场景检出攻击事件数/该场景攻击事件总数。AP和ROC-AUC按分数排序计算，不能仅凭混淆矩阵重算。",
  ]);
  section(
    "三、按攻击场景的最终表现（场景图数据）",
    ["攻击场景", "运行数", "事件总数", "检出事件数", "TP", "FP", "FN", "TN", "精确率", "逐行召回率", "F1", "误报率", "事件覆盖率", "AP", "ROC-AUC"],
    scenarioRows, [28, 12, 12, 14, 12, 12, 12, 14, 12, 14, 12, 12, 14, 12, 12],
    { B: "#,##0", C: "#,##0", D: "#,##0", E: "#,##0", F: "#,##0", G: "#,##0", H: "#,##0", I: "0.0000", J: "0.0000", K: "0.0000", L: "0.0000", M: "0.0000", N: "0.0000", O: "0.0000" },
  );

  const wins = getRows("csv_final_v6_scenario_wins");
  const orderedWins = wins.rows.slice().sort((a, b) => String(a[wins.index.scenario]).localeCompare(String(b[wins.index.scenario])));
  const winTotal = orderedWins.reduce((n, r) => n + Number(r[wins.index.wins] || 0), 0);
  const comparisonTotal = orderedWins.reduce((n, r) => n + Number(r[wins.index.total_baselines] || 0), 0);
  const winRows = orderedWins.map((r) => [
    r[wins.index.scenario], r[wins.index.candidate], r[wins.index.wins],
    r[wins.index.total_baselines], r[wins.index.win_share],
      r[wins.index.passes_single_scenario_70pct] === null
        ? null
        : (r[wins.index.passes_single_scenario_70pct] ? "是" : "否"),
  ]);
  winRows.push([
    "全局比较比例", "trust_shape_union", winTotal, comparisonTotal,
    winTotal / comparisonTotal, null,
  ]);
  formulaRows.push([
    "四、场景胜负比例",
    "每场景一行；另有一行全局汇总",
    "final_v6_scenario_wins.csv；trust_shape_union对11个V6非Trust基线；全局分母为9个场景×11个基线=99项。",
    `单场景胜率=胜场数/11，胜场比较基于事件覆盖率。全局比较比例=Σ场景胜场数/Σ场景比较数=${winTotal}/${comparisonTotal}=73.7%。它是比较胜率，不是性能指标；不代表每个场景领先或全面领先。单场景70%通过标记按冻结阈值规则计算。`,
  ]);
  section(
    "四、场景胜负比例（比较统计，不是性能指标）",
    ["攻击场景", "候选方法", "胜场数", "比较基线数", "胜率", "单场景是否达到70%"],
    winRows, [28, 28, 12, 14, 12, 20],
    { C: "#,##0", D: "#,##0", E: "0.0%" },
  );

  const ci = getRows("csv_final_v6_station_cluster_ci_corrected");
  const ciRows = ci.rows
    .filter((r) => r[ci.index.metric] === "event_coverage" && r[ci.index.candidate] === "trust_shape_union")
    .sort((a, b) => Number(a[ci.index.difference]) - Number(b[ci.index.difference]))
    .map((r) => [
      r[ci.index.candidate], r[ci.index.baseline],
      r[ci.index.metric] === "event_coverage" ? "事件覆盖率" : r[ci.index.metric],
      r[ci.index.difference], r[ci.index.ci95_low], r[ci.index.ci95_high],
      r[ci.index.p_two_sided], r[ci.index.bh_q], r[ci.index.provider_count],
      r[ci.index.station_count], r[ci.index.iterations],
      r[ci.index.resampling] === "providers_with_replacement_then_stations_within_provider_with_replacement"
        ? "先有放回重抽提供方，再在提供方内有放回重抽站点"
        : r[ci.index.resampling],
    ]);
  formulaRows.push([
    "五、事件覆盖率差值与站点聚类区间",
    "candidate=trust_shape_union；metric=event_coverage；对11个基线各一行",
    "final_v6_station_cluster_ci_corrected.csv；筛选trust_shape_union的事件覆盖率比较；4个提供方、42个站点、10,000次抽样。",
    "差值=候选方法事件覆盖率−基线事件覆盖率。95%区间来自分层聚类自助法：先有放回重抽提供方，再在提供方内有放回重抽站点。BH q值按预设的11项比较族校正。区间仅适用于本研究观察到的提供方/站点条件，不证明对更大总体的推广。",
  ]);
  section(
    "五、事件覆盖率差值与站点聚类不确定性（森林图数据）",
    ["候选方法", "对照基线", "指标", "差值", "95%区间下限", "95%区间上限", "双侧p值", "BH校正q值", "提供方数", "站点数", "重抽样次数", "重抽样方法"],
    ciRows, [28, 30, 18, 12, 14, 14, 12, 14, 12, 12, 14, 65],
    { D: "0.0000", E: "0.0000", F: "0.0000", G: "0.0000", H: "0.0000", I: "#,##0", J: "#,##0", K: "#,##0" },
  );

  const transfer = getRows("csv_final_external_transfer");
  const transferRows = transfer.rows.map((r) => [
    r[transfer.index.source],
    ({
      "locked external transfer": "锁定外部迁移",
      "station holdout locked transfer": "站点留出锁定迁移",
      "factor holdout exploratory audit": "因素留出探索性审计",
      "third-station holdout exploratory audit": "第三站点留出探索性审计",
    })[r[transfer.index.scope]] || r[transfer.index.scope],
    ({
      PASS_LOCKED_EXTERNAL_TRANSFER: "锁定外部迁移完成",
      COMPLETED_SMHI_LOCKED_TRANSFER: "SMHI锁定迁移完成",
      FAIL_SMHI_FACTOR_HOLDOUT_FACTOR_GATE: "未通过SMHI因素留出门槛",
    })[r[transfer.index.status]] || r[transfer.index.status],
    r[transfer.index.trust_f1], r[transfer.index.best_baseline],
    r[transfer.index.best_baseline_f1], r[transfer.index.trust_event_coverage],
    r[transfer.index.best_baseline_event_coverage],
    ({
      "One-Class SVM is stronger; FMI does not support provider-independent superiority.": "One-Class SVM表现更强；FMI结果不支持提供方无关的普遍优势。",
      "transfer_gate_status=NOT_PASSED; primary_pass_count=8/22": "迁移门槛未通过；主要比较通过数为8/22。",
      "scenario_win_share=0.5455; unseen_combination_event_coverage=0.0": "场景比较胜率为0.5455；未见组合事件覆盖率为0。",
      "scenario_win_share=0.5657; unseen_combination_event_coverage=0.1": "场景比较胜率为0.5657；未见组合事件覆盖率为0.1。",
    })[r[transfer.index.gate_or_boundary]] || r[transfer.index.gate_or_boundary],
  ]);
  formulaRows.push([
    "六、外部迁移与负面边界",
    "FMI、SMHI分别一行；探索性审计单独保留",
    "final_external_transfer.csv；逐来源逐范围保留，未与V6四提供方主结果合并。",
    "保留来源表中的独立汇总值。F1=2×精确率×召回率/(精确率+召回率)；事件覆盖率=检出事件数/攻击事件总数。不同来源的协议与范围不相同，因此不可将数值合并为总分。缺失项保留空值。",
  ]);
  section(
    "六、外部迁移与负面边界（单独报告，不并入V6）",
    ["来源", "范围", "状态", "Trust F1", "最佳基线", "最佳基线F1", "Trust事件覆盖率", "最佳基线事件覆盖率", "结果边界"],
    transferRows, [14, 38, 40, 12, 24, 16, 18, 22, 85],
    { D: "0.0000", F: "0.0000", G: "0.0000", H: "0.0000" },
  );

  const recovery = getRows("csv_final_online_offline_recovery");
  const recoveryRows = recovery.rows.map((r) => [
    r[recovery.index.attack_type], r[recovery.index.runs],
    r[recovery.index.online_mae], r[recovery.index.offline_mae],
      r[recovery.index.causal_penalty_mae], r[recovery.index.online_rmse],
    r[recovery.index.offline_rmse], r[recovery.index.causal_penalty_rmse],
      r[recovery.index.online_equals_offline],
  ]);
  formulaRows.push([
    "七、在线因果恢复与离线参考",
    "按攻击类型；6类回环测试攻击",
    "final_online_offline_recovery.csv；授权隔离回环测试台；按攻击类型、每类5次运行。与V6批处理检测性能分开。",
    "MAE因果惩罚=在线MAE−离线MAE；RMSE因果惩罚=在线RMSE−离线RMSE。在线恢复仅使用决策时可得信息，离线双向插值使用未来信息，因此离线值只是参考上限，不能表述为在线能力。",
  ]);
  section(
    "七、在线因果恢复与离线参考（恢复差值图数据）",
    ["攻击类型", "运行数", "在线MAE", "离线MAE", "MAE因果惩罚", "在线RMSE", "离线RMSE", "RMSE因果惩罚", "在线等同离线"],
    recoveryRows, [24, 12, 14, 14, 18, 14, 14, 18, 18],
    { B: "#,##0", C: "0.0000", D: "0.0000", E: "0.0000", F: "0.0000", G: "0.0000", H: "0.0000" },
  );

  sheet.getRange(`A1:A${row}`).format.columnWidth = 30;
  sheet.getRange(`B1:B${row}`).format.columnWidth = 30;
  sheet.getRange(`C1:C${row}`).format.columnWidth = 20;
  sheet.getRange(`D1:V${row}`).format.columnWidth = 16;
  sheet.freezePanes.freezeRows(2);
  addAnnotation(sheet, "A1", "论文直接制表/作图的最终数据集中页。所有数值来自权威V6修正表或独立补充结果表；来源列给出文件和筛选条件，公式列给出分子分母或聚合方式。不同估计目标、外部迁移、回环运行时和恢复结果不可拼接。V9未来时间窗未评估，阶段500仅为探索性证据，均未纳入。");
  const formulaSheet = workbook.worksheets.add(safeSheetName("公式与来源", usedNames));
  formulaSheet.showGridLines = false;
  const formulaHeaders = ["对应数据表", "数据范围/筛选", "权威来源", "计算公式、汇总规则与解释"];
  const formulaLast = writeRows(formulaSheet, formulaHeaders, formulaRows, 5000);
  formulaSheet.getRange(`A1:A${formulaLast}`).format.columnWidth = 34;
  formulaSheet.getRange(`B1:B${formulaLast}`).format.columnWidth = 56;
  formulaSheet.getRange(`C1:C${formulaLast}`).format.columnWidth = 80;
  formulaSheet.getRange(`D1:D${formulaLast}`).format.columnWidth = 120;
  formulaSheet.getRange(`A2:D${formulaLast}`).format.wrapText = true;
  formulaSheet.getRange(`A2:D${formulaLast}`).format.verticalAlignment = "center";
  for (let r = 2; r <= formulaLast; r++) formulaSheet.getRange(`A${r}:D${r}`).format.rowHeight = 64;
  addAnnotation(formulaSheet, "A1", "每一行对应“论文直用数据”中的同名编号结果表。来源文件和筛选条件可用于定位原始权威数据，公式列说明如何复算或解释；AP、ROC-AUC、bootstrap区间等需要分数排序或重抽样记录，不能仅凭表面汇总计数重建。");
  created.push({ tableName: "derived_paper_ready_formulas", sheetName: formulaSheet.name, rows: formulaRows.length, columns: formulaHeaders.length });
  return { tableName: "derived_paper_ready_data", sheetName: sheet.name, rows: dataRowCount, columns: "分区表" };
}

const usedNames = new Set();
const created = [];

const readme = workbook.worksheets.add(sheetNameMap.README);
usedNames.add(sheetNameMap.README.toLowerCase());
const readmeRows = [
  ["TrustMeteo-QC 实验数据总表", "说明"],
  ["用途", "将最终证据库中的结构化实验结果、审计记录、审计报告及论文材料汇总到一个工作簿中，便于查阅和后续写作。"],
  ["整理日期", "2026-10-11"],
  ["主要证据", "V6 修正后的结果是主证据。请区分按计数合并、提供方等权、提供方与随机种子宏平均、站点等权这几种统计口径，它们回答的问题不同。"],
  ["结论范围", "目前结果支持：在本实验的模拟攻击方案中，方法在多数比较中表现较好，但也存在混合结果和失败场景。结果不能证明生产环境全面领先，也不能证明对未测试提供方或站点普遍有效。"],
  ["V6 核心结果", "trust_shape_union 按计数合并：F1=0.228958，事件覆盖率=0.348256，误报率=0.008413。提供方与随机种子宏平均：F1=0.220268，事件覆盖率=0.355959，误报率=0.009210。两组数字属于不同统计口径，请在“核心结果”表中分别查看。"],
  ["探索性结果", "阶段500只用于探索和敏感性分析，不能替代 V6 主结果，也不能写成确认性结论。"],
  ["未评估内容", "V9 未来时间窗尚未准备好，也没有进行评估，因此不属于性能结果。"],
  ["方法限制", "V6 使用居中滚动窗口，因此属于批处理评估。在线测试台结果是另一组证据。在线因果恢复与离线双向插值不能视为同一能力。"],
  ["数据完整性", "84 张 CSV 数据表、审计 JSON、审计报告和论文材料已放入工作表。查看“文件清单”可查原文件路径、哈希值和是否已嵌入。"],
  ["中文说明", "主要数据表的列标题同时显示原始英文名和中文释义。关键指标的定义写在对应列标题批注中；每张数据表的 A1 批注说明证据类别和解释边界。"],
  ["原始材料", "论文、补充材料及审计原文保留原始文字，避免改写来源文件。Parquet 原始分数文件和图形文件只在“文件清单”中记录路径与哈希，没有展开到工作表。"],
];
writeRows(readme, readmeRows[0], readmeRows.slice(1), 5000);
readme.getRange(`A1:A${readmeRows.length}`).format.columnWidth = 28;
readme.getRange(`B1:B${readmeRows.length}`).format.columnWidth = 118;
readme.getRange(`B2:B${readmeRows.length}`).format.wrapText = true;
readme.getRange(`A2:B${readmeRows.length}`).format.verticalAlignment = "center";
for (let r = 2; r <= readmeRows.length; r++) readme.getRange(`A${r}:B${r}`).format.rowHeight = 42;
addAnnotation(readme, "A1", "先阅读本页，再查看“核心结果”。论文写作所用的权威数据表见“论文引用索引”。");
created.push({ sheetName: readme.name, rows: readmeRows.length - 1, columns: 2 });

const paperReady = addPaperReadySheet(usedNames);
created.push(paperReady);
const map = writeDataSheet("paper_source_map", sheetNameMap.Paper_Source_Map, usedNames);
created.push(map);
const catalog = writeDataSheet("artifact_catalog", sheetNameMap.Artifact_Catalog, usedNames);
created.push(catalog);
{
  const { columns, rows } = readTable("csv_final_v6_aggregation_levels_corrected");
  const methodIndex = columns.indexOf("method");
  const aggregationIndex = columns.indexOf("aggregation");
  const estimandIndex = columns.indexOf("estimand");
  const notesIndex = columns.indexOf("notes");
  const selected = rows.filter((row) => row[methodIndex] === "trust_shape_union").map((row) => {
    const out = [...row];
    out[aggregationIndex] = valueZh[out[aggregationIndex]] || out[aggregationIndex];
    out[estimandIndex] = valueZh[out[estimandIndex]] || out[estimandIndex];
    if (notesIndex >= 0) out[notesIndex] = primaryNoteZh[out[notesIndex]] || out[notesIndex];
    return out;
  });
  const sheet = workbook.worksheets.add(safeSheetName(sheetNameMap.Primary_Metrics, usedNames));
  const primaryHeaders = columns.map(bilingualHeader);
  const lastRow = writeRows(sheet, primaryHeaders, selected, 5000);
  formatByHeader(sheet, columns, lastRow);
  addAnnotation(
    sheet,
    "A1",
    "V6 主结果：展示 trust_shape_union 在不同统计口径下的结果。按计数合并代表合并样本计数后计算；宏平均、提供方等权、站点等权代表不同加权问题。本表不支持“全面领先”的结论。",
  );
  created.push({
    tableName: "derived_primary_metrics_view",
    sheetName: sheet.name,
    rows: selected.length,
    columns: columns.length,
  });
}

const priority = [
  "csv_final_v6_aggregation_levels_corrected",
  "csv_final_v6_method_summary",
  "csv_final_v6_scenario_wins",
  "csv_final_v6_scenario_ci_corrected",
  "csv_final_v6_station_cluster_ci_corrected",
  "csv_final_v6_union_decision_provenance",
  "csv_final_v6_scenario_metrics",
  "csv_final_v6_scenario_aggregate",
  "csv_final_v6_station",
  "csv_final_v6_event_lengths",
  "csv_final_v6_selection",
  "csv_final_v6_detail",
  "csv_final_v6_detail_counts_reconstructed",
  "csv_final_v6_aggregation_levels",
  "csv_final_external_transfer",
  "csv_final_negative_boundaries",
  "csv_final_evidence_hierarchy",
  "csv_final_method_implementation_boundary_corrected",
  "csv_final_baseline_resources_corrected",
  "csv_final_trust_resources_corrected",
  "csv_final_testbed_resources_corrected",
  "csv_final_online_detection",
  "csv_final_online_offline_recovery",
  "csv_final_operating_points",
];
const allTableNames = db.prepare(
  "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'csv_%' ORDER BY name",
).all().map((x) => x.name);
const orderedTables = [...priority.filter((n) => allTableNames.includes(n)), ...allTableNames.filter((n) => !priority.includes(n))];
const workbookIndex = [
  ["原始数据表名", "工作表名称", "数据行数", "字段数", "证据类别", "阅读说明"],
  ["derived_paper_ready_data", paperReady.sheetName, paperReady.rows, paperReady.columns, "论文直接使用的终值", "论文表图数据与每个指标的来源、筛选条件及计算公式。不同统计口径和证据层级分开列示。"],
  ["derived_paper_ready_formulas", "公式与来源", 7, 4, "论文数据复核依据", "按结果表说明数据来源、筛选条件、计算公式和统计口径。"],
];
for (const tableName of orderedTables) {
  const result = writeDataSheet(tableName, displayTableName(tableName), usedNames);
  created.push(result);
  const cls = tableName.startsWith("csv_stage500_")
    ? "仅作探索性分析"
    : tableName.startsWith("csv_stage")
      ? "冻结后探索/敏感性分析"
      : tableName.startsWith("csv_final_v6_")
        ? "V6证据，核对统计口径"
        : "最终补充证据";
  workbookIndex.push([
    tableName,
    result.sheetName,
    result.rows,
    result.columns,
    cls,
    sheetDescription(tableName),
  ]);
}
{
  const sheet = workbook.worksheets.add(safeSheetName(sheetNameMap.Workbook_Index, usedNames));
  const lastRow = writeRows(sheet, workbookIndex[0], workbookIndex.slice(1), 5000);
  formatByHeader(sheet, workbookIndex[0], lastRow);
  sheet.getRange(`A1:A${lastRow}`).format.columnWidth = 52;
  sheet.getRange(`B1:B${lastRow}`).format.columnWidth = 34;
  sheet.getRange(`C1:D${lastRow}`).format.columnWidth = 12;
  sheet.getRange(`E1:E${lastRow}`).format.columnWidth = 42;
  sheet.getRange(`F1:F${lastRow}`).format.columnWidth = 100;
  sheet.getRange(`F2:F${lastRow}`).format.wrapText = true;
  addAnnotation(sheet, "A1", "此表将数据库中的原始表名对应到工作簿中的中文工作表名称，并标明每张表的证据类别和阅读限制。");
  created.push({
    tableName: "derived_workbook_index",
    sheetName: sheet.name,
    rows: workbookIndex.length - 1,
    columns: workbookIndex[0].length,
  });
}

{
  const fields = new Set();
  for (const tableName of allTableNames) {
    const quoted = `"${tableName.replaceAll('"', '""')}"`;
    for (const column of db.prepare(`PRAGMA table_info(${quoted})`).all()) {
      fields.add(column.name);
    }
  }
  const fieldRows = [...fields].sort().map((field) => [
    field,
    translateField(field),
    fieldNotes[field] || "中文释义按字段名称拆分翻译。字段原名保留，具体分母、统计范围或单位请结合所在工作表及原始方法说明判断。",
  ]);
  for (const [term, translation] of Object.entries(valueZh)) {
    fieldRows.push([term, translation, "这是结果表中的分类值或证据类别名称，不是数据列名。"]);
  }
  const sheet = workbook.worksheets.add(safeSheetName(sheetNameMap.Data_Dictionary, usedNames));
  const headers = ["原始字段名或结果术语", "中文释义", "定义与阅读提示"];
  const lastRow = writeRows(sheet, headers, fieldRows, 5000);
  sheet.getRange(`A1:A${lastRow}`).format.columnWidth = 38;
  sheet.getRange(`B1:B${lastRow}`).format.columnWidth = 42;
  sheet.getRange(`C1:C${lastRow}`).format.columnWidth = 110;
  sheet.getRange(`C2:C${lastRow}`).format.wrapText = true;
  addAnnotation(sheet, "A1", "本表收录全部结果表中的唯一字段名，并列出常见结果分类词。结果数据列保留原始英文名，主结果表同时显示中文释义。不能仅凭字段名确定的定义会提醒结合所在工作表和方法说明核对。");
  created.push({
    tableName: "derived_data_dictionary",
    sheetName: sheet.name,
    rows: fieldRows.length,
    columns: headers.length,
  });
}

created.push(writeTextSheet("json_artifacts", sheetNameMap.JSON_Audits, usedNames));
created.push(writeTextSheet("report_text", sheetNameMap.Audit_Reports, usedNames));

const manuscriptRows = [];
for (const rel of [
  "manuscript_final_20261009.md",
  "supplementary_final_20261009.md",
  "submission_materials_final_20261009.md",
  "README.md",
]) {
  const p = path.join(finalDir, rel);
  const content = await fs.readFile(p, "utf8");
  const hash = crypto.createHash("sha256").update(await fs.readFile(p)).digest("hex").toUpperCase();
  manuscriptRows.push([path.basename(p), path.relative(path.dirname(finalDir), p).replaceAll("\\", "/"), hash, content]);
}
created.push(writeOriginalTextRows(manuscriptRows, sheetNameMap.Manuscript_Materials, usedNames));

workbook.recalculate();

const keySheetName = created.find(
  (sheet) => sheet.tableName === "csv_final_v6_aggregation_levels_corrected",
)?.sheetName;
const keySheet = workbook.worksheets.getItem(keySheetName);
console.log("KEY_VALUES", JSON.stringify(keySheet.getRange("A1:W8").values));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);

const fileStat = await fs.stat(outputPath);
console.log(JSON.stringify({
  status: "PASS_XLSX_EXPORTED",
  output: outputPath,
  bytes: fileStat.size,
  tableCount: allTableNames.length,
  sheets: created.length,
  sheetSummary: created,
}, null, 2));

db.close();
