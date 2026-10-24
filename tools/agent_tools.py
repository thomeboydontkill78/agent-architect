import csv
import json
import logging
from pathlib import Path

from langchain_core.tools import tool

from config.settings import settings
from services.rag_service import RagSummarizeService

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


_rag_service = None
def get_rag_service() -> RagSummarizeService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


_external_data: dict | None = None
def get_external_data() -> dict:
    global _external_data
    if _external_data is not None:
        return _external_data

    data: dict = {}
    file_path = _resolve_path(settings.EXTERNAL_DATA_PATH)

    if not file_path.exists():
        logger.warning("外部数据文件不存在：%s", file_path)
        _external_data = data
        return _external_data

    try:
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                logger.warning("外部数据文件为空：%s", file_path)
                _external_data = data
                return _external_data
            for row in reader:
                if len(row) < 6:
                    continue
                uid, feature, eff, cons, comp, month = [c.strip() for c in row[:6]]
                if not uid or not month:
                    continue
                data.setdefault(uid, {})[month] = {
                    "特征": feature, "效率": eff, "耗材": cons, "对比": comp,
                }
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        logger.exception("读取外部数据失败：%s", e)

    _external_data = data
    return _external_data


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return get_rag_service().rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26摄氏度，湿度50%"  # mock


@tool(description="获取用户所在城市名称")
def get_user_location() -> str:
    return "深圳"  # mock


@tool(description="获取用户ID")
def get_user_id() -> str:
    data = get_external_data()
    return "" if not data else next(iter(data.keys()))


@tool(description="获取当前月份")
def get_current_month() -> str:
    return "2025-06"  # mock


@tool(description="从外部系统获取指定用户指定月份的使用记录，未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    rec = get_external_data().get(user_id, {}).get(month)
    if rec is None:
        logger.warning("[fetch_external_data] 未检索到 user=%s month=%s", user_id, month)
        return ""
    return json.dumps(rec, ensure_ascii=False)


@tool(description="无入参，调用后触发中间件切换报告提示词")
def fill_context_for_report() -> str:
    return "fill_context_for_report已调用"
