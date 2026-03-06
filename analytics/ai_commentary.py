"""AI-powered commentary for analytical results.

Uses Anthropic Claude API to generate interpretive text
based on numerical analysis results.
"""

import hashlib
import json
import logging
import os

import streamlit as st

logger = logging.getLogger(__name__)


def _get_api_key() -> str | None:
    """Get Anthropic API key from env or Streamlit secrets."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    return key or None


def _hash_data(data: dict) -> str:
    """Create a stable hash of input data for caching."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


@st.cache_data(ttl=3600, show_spinner=False)
def _call_claude(prompt: str, data_hash: str) -> str:
    """Call Claude API with caching. data_hash is used for cache key."""
    api_key = _get_api_key()
    if not api_key:
        return ""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except ImportError:
        logger.warning("anthropic package not installed")
        return ""
    except Exception as e:
        logger.warning("Claude API call failed: %s", e)
        return ""


def generate_factor_summary_commentary(
    stability_data: list[dict],
    class_name: str,
    factor_labels: dict[str, str],
) -> str:
    """Generate AI commentary for factor validation cards."""
    if not stability_data:
        return ""

    # Build a concise data summary for the prompt
    factors_info = []
    for f in stability_data:
        fname = factor_labels.get(f["factor"], f["factor"])
        factors_info.append(
            f"{fname}: R²={f['r2']:.4f}, p={f['pvalue']:.4f}, "
            f"OOS R²={f['oos_r2']:.4f}, статус={f['status']}"
        )

    prompt = f"""Ты финансовый аналитик. Дай краткий (3-5 предложений) комментарий по результатам
однофакторного анализа влияния макро- и рыночных факторов на торговые обороты класса
инструментов "{class_name}" на Московской бирже.

Результаты по каждому фактору (R² — доля объяснённой дисперсии, p-value — статистическая
значимость, OOS R² — предсказательная сила на новых данных, статус — green/yellow/red):

{chr(10).join(factors_info)}

Укажи:
1. Какие факторы наиболее значимы и почему
2. Какие факторы слабо влияют
3. Общий вывод о предсказуемости оборотов этого класса

Пиши на русском, кратко, профессионально. Не повторяй числа — интерпретируй их."""

    data_hash = _hash_data({"type": "factor_summary", "class": class_name,
                             "data": stability_data})
    return _call_claude(prompt, data_hash)


def generate_correlation_commentary(
    corr_matrix_data: dict,
    method: str,
    class_labels: dict[str, str],
    factor_labels: dict[str, str],
) -> str:
    """Generate AI commentary for correlation heatmap."""
    if not corr_matrix_data:
        return ""

    # Find top positive and negative correlations
    pairs = []
    for cls, factors in corr_matrix_data.items():
        cls_name = class_labels.get(cls, cls)
        for fac, val in factors.items():
            fac_name = factor_labels.get(fac, fac)
            pairs.append((cls_name, fac_name, val))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = pairs[:10]

    pairs_text = "\n".join(
        f"  {cls} ~ {fac}: r={val:+.3f}"
        for cls, fac, val in top_pairs
    )

    prompt = f"""Ты финансовый аналитик. Дай краткий (3-5 предложений) комментарий по матрице
корреляций ({method}) между дневными изменениями торговых оборотов классов инструментов
MOEX и изменениями макро/рыночных факторов.

Топ-10 пар по абсолютной корреляции:
{pairs_text}

Интерпретируй:
1. Какие связи наиболее сильные и что они означают экономически
2. Есть ли неожиданные закономерности
3. Общая картина зависимостей оборотов от факторов

Пиши на русском, кратко, профессионально. Укажи направление связи (прямая/обратная)."""

    data_hash = _hash_data({"type": "correlation", "method": method,
                             "pairs": [(c, f, round(v, 4)) for c, f, v in top_pairs]})
    return _call_claude(prompt, data_hash)


def generate_regression_commentary(
    r2: float,
    adj_r2: float,
    n_obs: int,
    coefficients: dict[str, float],
    pvalues: dict[str, float],
    class_name: str,
    factor_labels: dict[str, str],
) -> str:
    """Generate AI commentary for regression results."""
    if not coefficients:
        return ""

    # Build factor summary
    factors_info = []
    for k in sorted(coefficients.keys(), key=lambda k: abs(coefficients[k]), reverse=True):
        pv = pvalues.get(k, 1.0)
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "н/з"
        fname = factor_labels.get(k, k)
        factors_info.append(f"  {fname}: β={coefficients[k]:+.4f}, p={pv:.4f} ({sig})")

    prompt = f"""Ты финансовый аналитик. Дай краткий (4-6 предложений) комментарий по результатам
многофакторной OLS-регрессии для дневных изменений оборотов класса "{class_name}" на MOEX.

Метрики модели:
  R² = {r2:.4f}, R² adj = {adj_r2:.4f}, наблюдений = {n_obs}

Стандартизированные коэффициенты (отсортированы по |β|):
{chr(10).join(factors_info)}

Интерпретируй:
1. Оцени качество модели по R² (какую долю вариации объясняет)
2. Какие факторы оказывают наибольшее влияние на обороты и в каком направлении
3. Какие факторы незначимы и можно исключить
4. Практический вывод для понимания динамики оборотов этого класса

Пиши на русском, кратко, профессионально. Дай практические выводы."""

    data_hash = _hash_data({"type": "regression", "class": class_name,
                             "r2": round(r2, 4), "adj_r2": round(adj_r2, 4),
                             "coefficients": {k: round(v, 4) for k, v in coefficients.items()}})
    return _call_claude(prompt, data_hash)
