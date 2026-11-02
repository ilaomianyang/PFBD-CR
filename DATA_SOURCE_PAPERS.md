# 数据源论文与官方出处

[English README](README.md) | [简体中文 README](README.zh-CN.md)

本文档回答一个具体问题：TrustMeteo-QC 每个数据源应当引用哪篇论文或哪份官方
数据说明。检索和 DOI 核对日期为 2026-11-02。

判断规则：

- 优先引用直接描述该数据产品、观测网或数据访问系统的论文。
- 如果只找到数据使用论文或一般背景论文，不把它冒充为数据描述论文。
- 如果当前接口或数据产品没有专门论文，明确写“未找到”，并列出权威官方文档。
- 论文元数据存在不代表已经逐篇阅读全文；正式投稿前仍需核对引用段落。

## 1. 汇总表

| 数据源 | 直接论文/数据说明 | DOI | 官方数据入口 | 当前用途 |
|---|---|---|---|---|
| NOAA ISD | Smith, Lott & Vose (2011) | `10.1175/2011BAMS3015.1` | <https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database> | V6 四提供方之一 |
| ERA5 | Hersbach et al. (2020) | `10.1002/qj.3803` | <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels> | 背景场和偏差订正 |
| DWD CDC | Kaspar, Kratzenstein & Kaiser-Weiss (2019) | `10.5194/asr-16-75-2019` | <https://opendata.dwd.de/climate_environment/CDC/> | 历史单提供方基准 |
| MeteoSwiss | Landl, Roulet & Calpini (2009) | 无 DOI | <https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html> | V6 四提供方之一 |
| KNMI | 未找到直接数据论文 | 无 | <https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens> | V6 四提供方之一 |
| ECCC/MSC | 未找到直接数据论文 | 无 | <https://api.weather.gc.ca/collections/climate-hourly> | V6 四提供方之一 |
| SWaT | Mathur & Tippenhauer (2016) | `10.1109/CySWater.2016.7469060` | <https://itrust.sutd.edu.sg/itrust-labs_datasets/> | 安全域补充实验 |
| USAD 代码 | Audibert et al. (2020) | `10.1145/3394486.3403392` | <https://github.com/manigalati/usad> | 第三方深度基线 |
| FMI | 未找到直接数据论文 | 无 | <https://en.ilmatieteenlaitos.fi/open-data> | 单机构锁定外部验证 |
| SMHI | 未找到直接数据论文 | 无 | <https://opendata-download-metobs.smhi.se/> | 单机构锁定外部验证 |
| MET Norway Frost | 未找到 Frost 专门论文 | 无 | <https://frost.met.no/> | 独立机构锁定迁移 |
| 日本 NOAA ISD 网络 | 同 NOAA ISD 论文 | `10.1175/2011BAMS3015.1` | <https://www.ncei.noaa.gov/data/global-hourly/access/> | 同提供方站点网络补充 |
| 六盘山本地观测 | 未找到公开论文 | 无 | 无公开 URL | 早期本地实验背景 |

“未找到”表示截至检索日期没有找到与当前所用数据产品直接匹配的同行评议数据论文。
它不等于该机构没有任何论文，也不等于可以用一篇不相关的论文替代官方数据说明。

## 2. 可直接引用的数据源论文

### NOAA ISD

Smith, A., Lott, N., & Vose, R. (2011). The Integrated Surface Database: Recent
Developments and Partnerships. *Bulletin of the American Meteorological
Society*, 92(6), 704-708. <https://doi.org/10.1175/2011BAMS3015.1>

适用性：直接介绍 NOAA/NCEI Integrated Surface Database。可用于 V6 的 NOAA ISD
数据以及日本站点网络补充实验。日本网络仍属于同一 NOAA provider，不能据此宣称
机构独立。

### ERA5

Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., et al.
(2020). The ERA5 Global Reanalysis. *Quarterly Journal of the Royal
Meteorological Society*, 146(730), 1999-2049.
<https://doi.org/10.1002/qj.3803>

适用性：直接介绍 ERA5 再分析产品。数据集本身还应引用 Copernicus 数据集 DOI
`10.24381/cds.adbb2d47`，并写明下载变量、区域、时间范围和访问日期。ERA5 是再分析
背景场，不是独立现场传感器真值。

### DWD CDC

Kaspar, F., Kratzenstein, F., & Kaiser-Weiss, A. K. (2019). Interactive Open
Access to Climate Observations from Germany. *Advances in Science and Research*,
16, 75-83. <https://doi.org/10.5194/asr-16-75-2019>

适用性：介绍 DWD 气候观测的开放访问和 CDC 门户。论文不替代具体站点归档文件、
变量说明、质量码和下载日期。

### MeteoSwiss SwissMetNet

Landl, B., Roulet, Y.-A., & Calpini, B. (2009). SwissMetNet: Operational
Quality Control on Raw Data of the New Automatic Meteorological Ground-Based
Network of Switzerland. EMS Annual Meeting Abstracts.
<https://ui.adsabs.harvard.edu/abs/2009ems..confE.453L/abstract>

适用性：直接涉及 SwissMetNet 原始数据业务质量控制，但没有 DOI，且属于会议摘要，
不是完整的数据描述论文。正式论文中应同时引用 MeteoSwiss OGD STAC 数据集页面：
<https://data.geo.admin.ch/api/stac/v0.9/collections/ch.meteoschweiz.ogd-smn>。

### SWaT

Mathur, A. P., & Tippenhauer, N. O. (2016). SWaT: A Water Treatment Testbed for
Research and Training on ICS Security. *2016 International Workshop on
Cyber-Physical Systems for Smart Water Networks*, 31-36.
<https://doi.org/10.1109/CySWater.2016.7469060>

适用性：直接介绍 SWaT 数据集和测试床。SWaT 不是气象生产网络，结果必须与气象
主证据分层。

### USAD

Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A. (2020).
USAD: UnSupervised Anomaly Detection on Multivariate Time Series.
*Proceedings of the 26th ACM SIGKDD International Conference on Knowledge
Discovery & Data Mining*, 3395-3404.
<https://doi.org/10.1145/3394486.3403392>

适用性：USAD 方法和公开代码的来源论文。USAD 论文不授予 SWaT 数据再分发权。

## 3. 没有直接数据论文时的官方引用

### KNMI

本项目使用 KNMI 小时地面气象观测。未找到专门描述当前 `uurgegevens` 文件和
Data Platform 接口的同行评议数据论文。建议在论文的数据来源表中写：

> Royal Netherlands Meteorological Institute (KNMI). Hourly meteorological
> observations of the Netherlands. Official data product and Data Platform,
> accessed [YYYY-MM-DD].

官方入口：

- <https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens>
- <https://dataplatform.knmi.nl/>
- <https://developer.dataplatform.knmi.nl/open-data-api>

项目记录许可为 CC BY 4.0。最终引用应写明具体站号、年份、变量和下载文件版本。

### ECCC/MSC

本项目使用 ECCC/MSC GeoMet `climate-hourly`。未找到专门描述该 GeoMet 集合的同行
评议数据论文。建议引用数据集名称、集合 ID、访问日期和官方手册：

> Environment and Climate Change Canada / Meteorological Service of Canada.
> Climate - Hourly Observations. GeoMet OGC API, accessed [YYYY-MM-DD].

官方入口：

- <https://api.weather.gc.ca/collections/climate-hourly>
- <https://api.weather.gc.ca/collections/climate-stations>
- <https://www.canada.ca/en/environment-climate-change/services/manuals/surface-weather-observations.html>

许可为 Open Government Licence - Canada：
<https://open.canada.ca/en/open-government-licence-canada>。

### FMI

本项目使用 FMI Open Data WFS。未找到专门描述该开放接口和当前站点提取协议的论文。
建议引用官方 Open Data 服务：

> Finnish Meteorological Institute. FMI Open Data. WFS observation service,
> accessed [YYYY-MM-DD].

官方入口：

- <https://en.ilmatieteenlaitos.fi/open-data>
- <https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0>

项目记录许可为 Creative Commons Attribution 4.0。论文中应写明查询参数、站点、
变量、时间范围和下载哈希。

### SMHI

本项目使用 SMHI Open Data meteorological observations。未找到专门描述当前 API
版本和所用观测产品的同行评议数据论文。建议引用官方 API：

> Swedish Meteorological and Hydrological Institute. Open Data Meteorological
> Observations. API version 1.0, accessed [YYYY-MM-DD].

官方入口：

- <https://opendata-download-metobs.smhi.se/>
- <https://opendata-download-metobs.smhi.se/api/version/1.0/>

许可和数据版本应以 SMHI 页面及下载响应中的元数据为准。SMHI 负结果必须保留。

### MET Norway Frost

未找到专门描述 Frost API 的同行评议数据论文。建议引用官方服务：

> Norwegian Meteorological Institute (MET Norway). Frost API. Observations and
> metadata service, accessed [YYYY-MM-DD].

官方入口：

- <https://frost.met.no/>
- <https://frost.met.no/swagger.json>

项目记录许可为 NLOD 2.0。论文中应写明 Frost source ID、element ID、质量码规则、
时间范围和下载哈希。

### 六盘山本地观测

六盘山本地观测没有仓库内可公开的论文或 URL。投稿时必须在数据可用性声明中写明
实际数据所有者、提供方式、许可或不能公开的原因。不能用 NOAA、ERA5 或一般气象
数据论文替代本地观测的来源说明。

## 4. 论文中的推荐写法

在 Data Availability 或数据来源表中，建议按以下结构组织：

| Provider | Product and citation | Sites/years | Variables | Access date | License |
|---|---|---|---|---|---|
| NOAA/NCEI | ISD; Smith et al. (2011) | Stations and years | Temperature | Date | NCEI terms |
| MeteoSwiss | OGD SMN; Landl et al. (2009) plus STAC | Stations and years | Temperature | Date | CC-BY |
| KNMI | Hourly observations; official API | Stations and years | Temperature | Date | CC BY 4.0 |
| ECCC/MSC | GeoMet climate-hourly | Stations and years | Temperature | Date | OGL Canada |
| Copernicus | ERA5; Hersbach et al. (2020) | Region and years | Variables | Date | CDS licence |

其余外部来源应单独成表，不能混入 V6 四提供方主估计。正式投稿前应补全实际站点、
年份、变量、访问日期、许可快照和文件哈希。

## 5. 检索限制

- 本检索使用 Crossref、OpenAlex 和官方数据页面核对论文题名、作者、年份和 DOI。
- “未找到直接数据论文”仅针对当前所用产品和接口，不排除机构发布的其他技术报告。
- MeteoSwiss 2009 条目为 EMS 会议摘要且无 DOI，引用时应标明文献类型。
- 未把一般使用该数据的论文当作数据来源论文，也未把方法论文当作数据集论文。
