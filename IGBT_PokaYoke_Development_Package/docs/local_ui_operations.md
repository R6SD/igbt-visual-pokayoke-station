# 本地影子工作台：安装、操作与数据维护

## 范围与安全边界

本服务仅在 `127.0.0.1` 监听，默认端口 `8765`。工作模式固定为 shadow，API 不提供 PLC、继电器、串口或其他生产联锁输出。屏幕上的 PASS 仅表示影子逻辑观察结果，不得作为生产放行信号。首次启动相机状态是 `NOT_CONNECTED`、`view_valid=false`；未知、失效或质量差一律不能显示为有效 PASS。

前端采用 React、Vite、Tailwind CSS v4 和 shadcn/ui（Radix primitives）。组件源代码保存在 `frontend/igbt-station-ui/src/components/ui/`，样式、布局与工位专用组件可在项目内维护。构建产物输出到 `web/dist/`，由 FastAPI 本地提供；运行时不访问 CDN、在线 API 或外部图标服务。Geist Variable 字体随前端包本地提供。需要 Node.js 22 或更新版本、Chrome/Edge 123 或更新版本。

现场仍为站立式人工目检，员工双手握住料管两端。界面把两端 grip zone 与中央 label contact ROI 分开；左右正常握持不会作为 C 事件。中央检测仅是 2D 接近代理证据，不能证明物理接触。

## 安装与启动（Windows PowerShell）

1. 安装受支持的 Python 3.11 或更新版本，并创建项目虚拟环境：

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements-ui.txt
   ```

2. 安装 Node.js 22 或更新版本，在前端目录安装锁定依赖并构建生产资源：

   ```powershell
   Set-Location .\frontend\igbt-station-ui
   npm ci
   npm run build
   Set-Location ..\..
   ```

   构建会生成 `web/dist/`。本地服务启动不需要 Node.js 常驻，也不需要前端开发服务器。修改前端代码后必须重新构建。

3. 从项目根目录启动：

   ```powershell
   .\scripts\start_local_ui.ps1
   ```

   首次启动会分别提示设置管理员令牌和运行时令牌。将其保存在受控的本机凭据库；不要提交到代码仓库。令牌只在进程环境与当前页面内存中使用。

4. 打开 `http://127.0.0.1:8765`。只通过本机回环地址访问。不要配置端口转发或反向代理。

可用 `-Port` 和 `-DataDirectory` 指定本机端口及数据位置；服务器仍然固定绑定 `127.0.0.1`。运行数据默认位于项目根目录 `data/`，也可设置 `IGBT_DATA_DIR`。

## 日常操作

- 总览页显示相机状态、FPS、标签状态、`view_valid` 与原因、左右握持、中央接触候选、FSM 状态和 C 连续计时。初始配置和运行时都不应把未知状态改成 absent。
- 管理员在“ROI 与阈值”页载入现场帧，分别框选 `tube_roi`、`label_roi`、`label_contact_roi` 和两端 `left_grip_zone` / `right_grip_zone`。画框坐标归一化到 0–1；相机设备 ID、分辨率、校准时间、ROI 配置版本及阈值会写入 SQLite 历史。
- 阈值必须从本站样本校准。空值表示未设置，不代表默认合格。任何 ROI 变更均新增版本，不覆写历史配置。
- “保存证据”支持初始标签、A、B、C 和 anomaly 图像。每次保存生成唯一文件、SHA256 和来源事件；服务不会覆盖已有原图。操作员字段只填匿名 ID，不填姓名或工号。
- 模型登记只保存名称、版本、SHA256、许可证状态和备注，不执行训练或自动加载模型。
- “现场画面”只显示最近证据关键帧，不是假称实时视频流；如果运行时尚未接入，画面和状态都保持未连接/未知。API 连接失败时 UI 会显示错误并保持 fail-closed。
- API 当前每 2.5 秒刷新状态。相机采集/FSM 进程需要使用独立 `IGBT_STATION_TOKEN` 调用 `POST /api/runtime/snapshot`；本包没有自动启动相机或检测模型的生产运行时。

## 备份

为保证 SQLite 与图像证据一致，建议停止服务后备份整个数据目录：

```powershell
Copy-Item -Recurse -Force .\data .\backup\data-YYYYMMDD-HHMM
```

如果使用自定义 `IGBT_DATA_DIR`，备份该目录而不是项目内 `data`。保留目录结构、`station.sqlite3`、`evidence/` 和 ACL；备份介质按现场图像隐私要求加密和限制访问。恢复前先验证备份文件数量及 SHA256（证据 API 返回 SHA256）。

## 恢复

1. 停止 FastAPI 进程。
2. 将备份中的完整数据目录复制回目标位置；不要只恢复数据库或只恢复图片。
3. 确认 `station.sqlite3` 和 `evidence/` 权限、大小及证据文件完整性。
4. 重新启动服务并检查 `/api/health`、配置历史、模型登记和周期证据列表。
5. 恢复后运行状态仍按 shadow 处理；摄像头运行时须重新连接并检查 `view_valid`，不得依据历史 PASS 自动放行。

## 故障排除

| 现象 | 检查和处理 |
|---|---|
| 浏览器显示 API 未连接 | 确认启动终端仍运行、端口未被占用、访问地址为 `127.0.0.1`；查看终端异常。 |
| 管理员操作返回 503 | 启动时没有设置 `IGBT_ADMIN_TOKEN`；停止服务后重新设置环境变量并启动。 |
| 管理员操作返回 401 | 令牌不匹配；重新输入启动时的管理员令牌。 |
| 保存 ROI 提示缺少 ROI | 覆盖两端时必须保存 tube、label、label contact 与 left/right grip 共五个 ROI。 |
| C 区域与握持区重叠 | 重新框选中央 ROI 和左右握持区；API 会拒绝明显重叠。 |
| 上传图片失败 | 只接受有效 JPEG/PNG，单张上限 10 MiB；确认文件扩展名、真实格式和周期 ID（字母、数字、下划线、连字符）。 |
| 状态一直是 NOT_CONNECTED / VIEW_INVALID | 本 UI 不采集相机。部署的采集适配器需发送完整 runtime snapshot；相机在线、有效视角、质量合格和 FSM 状态缺一不可。 |
| SQLite busy 或磁盘写入错误 | 确认只有一个服务进程写入该数据目录，磁盘空间与目录权限正常；安全停止后再备份或维护。 |

## 运维限制

- 不要通过 LAN、VPN、隧道或代理暴露服务。
- 不要把 token、原始图像、证据、SQLite 数据库或现场样本提交到版本库。
- 不要把 UI 的 PASS 当作联锁、质量放行或合规验证。
- SQLite/证据数据保留期限、匿名化和访问权限应按工厂数据治理制度设定；当前版本不自动清理数据。
