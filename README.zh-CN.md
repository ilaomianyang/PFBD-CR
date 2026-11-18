# PFBD-CR

[English](README.md) | **简体中文**

PFBD-CR 是面向气象传感器观测流的物理约束、虚警保界防御与因果恢复研究代码。
本仓库只发布代码、冻结协议和可独立运行的核心测试，不重新分发原始观测、结果表、
模型权重、测试床日志或未完成手稿。

历史上该代码曾以 TrustMeteo-QC 命名发布，现项目名与仓库名统一为 PFBD-CR。

## 当前实验边界

- 冻结 V6 主证据使用 ECCC、KNMI、MeteoSwiss 和 NOAA ISD 四个提供方，
  包含 9 类透明受控攻击、5 个固定随机种子和 11 个非 Trust 基线。
- 攻击是在真实气象观测背景上按冻结规则注入的，不是生产事故或第三方攻击日志。
- V6 的 Hampel 分量使用居中滑动窗口，因此属于包含未来信息的批处理评价，
  不能描述为严格因果在线检测。
- V9 未来窗口实验没有完成评价，不构成性能证据；Stage500 也不能替代冻结 V6。
- DWD 是历史单提供方基准，不与 V6 四提供方主估计合并。
- FMI、SMHI、MET Norway、日本 NOAA ISD 站点网络和 SWaT 必须单独报告。
  混合、负向或未通过门槛的结果应保留，不能改写成“全面领先”。
- 四个提供方支持有限范围的跨机构证据，不证明机构无关、站点无关或生产环境普适性。

可引用的数据源论文和官方产品说明见
[DATA_SOURCE_PAPERS.md](DATA_SOURCE_PAPERS.md)，BibTeX 条目见
[DATA_SOURCE_REFERENCES.bib](DATA_SOURCE_REFERENCES.bib)。

## 数据源论文的检索结论

| 数据源 | 是否有直接论文 | 论文或官方引用 |
|---|---|---|
| NOAA ISD | 有 | Smith et al. (2011)，DOI `10.1175/2011BAMS3015.1` |
| ERA5 | 有 | Hersbach et al. (2020)，DOI `10.1002/qj.3803` |
| DWD CDC | 有 | Kaspar et al. (2019)，DOI `10.5194/asr-16-75-2019` |
| MeteoSwiss SwissMetNet | 有早期会议论文 | Landl et al. (2009)，无 DOI，另有官方 STAC 说明 |
| SWaT | 有 | Mathur & Tippenhauer (2016)，DOI `10.1109/CySWater.2016.7469060` |
| USAD 代码 | 有 | Audibert et al. (2020)，DOI `10.1145/3394486.3403392` |
| KNMI | 未找到直接数据论文 | 引用 KNMI 官方小时观测页面和 Data Platform 文档 |
| ECCC | 未找到直接数据论文 | 引用 ECCC GeoMet `climate-hourly` 和官方观测手册 |
| FMI | 未找到直接数据论文 | 引用 FMI Open Data 和 WFS 说明 |
| SMHI | 未找到直接数据论文 | 引用 SMHI Open Data API 说明 |
| MET Norway | 未找到 Frost 数据论文 | 引用 Frost API 和 Swagger 文档 |
| 六盘山本地观测 | 无公开论文和公开 URL | 需要数据所有者确认来源与再分发许可 |

检索结果只能说“截至 2026-11-02 未找到专门描述该当前接口/数据产品的论文”，
不能据此断言绝对不存在其他论文。

## 仓库结构

```text
src/                        检测、融合、恢复、评价和审计代码
config/                     历史协议配置
confirmatory_package/       V4/V6 冻结协议和访问记录
scripts/                    证据表和工作簿构建工具
tests/                      不依赖原始数据的核心测试
vendor/USAD/                第三方 USAD 代码及其 BSD 许可证
requirements.txt            已验证的核心 Python 依赖
requirements-optional.txt   ERA5 下载的可选依赖
DATA_SOURCE_PAPERS.md       数据源论文、官方入口和证据边界
DATA_SOURCE_REFERENCES.bib  数据源论文的 BibTeX 条目
```

源码保留实验阶段编号，因为编号对应该脚本在实验链中的顺序和用途。高阶段脚本通常
依赖外部观测、协议或冻结中间产物，不属于最小核心测试路径。

## 环境安装

项目记录的发布环境是 Windows 和 Python 3.12。建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 核心测试

```powershell
python tests\run_tests.py
```

这组测试检查检测器接口、单站空间离群响应、验证集独立校准，以及恢复过程不能改写
未被检测器标记的正常行。

## 复现入口

冻结 V6 证据重建入口为：

```powershell
python src\stage470_final_v6_submission_rebuild.py
```

后续汇总、不确定性、来源追踪、资源边界和包级审计由 `stage478` 至 `stage485`
脚本实现。运行前需要从官方来源取得观测数据并准备冻结中间产物。协议 JSON 中的历史
路径是来源记录；在其他机器运行时，应复制协议并调整本地路径，不应修改已发布协议来
适配自己的目录。

## 数据与凭据

原始观测受各提供方许可约束。不要把数据、提供方凭据、`.secrets`、Cookie 或访问
令牌提交到仓库。ERA5 下载需要用户自己的 Copernicus CDS 配置；MET Norway Frost
需要用户自己的客户端凭据。

## 引用与许可

软件引用元数据见 [CITATION.cff](CITATION.cff)。关联论文尚未正式发表，因此当前
没有可引用的论文 DOI；发表后应补充真实作者、单位、卷期、页码和 DOI。

本仓库自有代码采用 MIT 许可证。`vendor/USAD/` 保留原始 BSD 许可证，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
