import os
import json
import time
from typing import Dict, Any, Optional, List

# Optional import for Groq
try:
    from groq import Groq
except Exception:
    Groq = None


class LLMAdvisor:
    """
    LLM-based Advisory Engine for personal finance.
    Supports Groq API backend with various models.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, concise, and practical personal finance advisor. "
        "Produce clear insights, actionable recommendations, and simple explanations that a non-expert can follow."
    )

    def __init__(self, backend: str = "groq", groq_model: str = "llama-3.3-70b-versatile", groq_api_key: Optional[str] = None):
        """
        backend: "groq" or "callable"
            - "groq": uses Groq API client.
            - "callable": you must provide a custom `call_llm_fn` to get_advice.
        groq_model: model name for Groq usage. Popular options:
            - "llama-3.3-70b-versatile" (recommended, fast and capable)
            - "llama-3.1-70b-versatile"
            - "mixtral-8x7b-32768"
            - "gemma2-9b-it"
        groq_api_key: optional override for environment variable.
        """
        self.backend = backend
        self.groq_model = groq_model
        
        # Initialize Groq client
        if backend == "groq":
            if Groq is None:
                raise RuntimeError("groq package not installed. Install with: pip install groq")
            
            api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set. Provide it via parameter or environment variable.")
            
            self.groq_client = Groq(api_key=api_key)

    # ---------------------------
    # PROMPT BUILDING
    # ---------------------------
    def _format_currency(self, value: float) -> str:
        try:
            return f"₹{value:,.0f}"
        except Exception:
            return str(value)

    def build_prompt(self, metrics: Dict[str, Any]) -> str:
        """
        Build a single string prompt summarizing the metrics.
        `metrics` should contain keys:
            - monthly_spending (number)
            - inflow (number)
            - savings (number)
            - highest_category (tuple(str, number))
            - unusual_spends (list of dicts: {"merchant":str,"amount":num,"note":str})
            - recurring_bills (list of str)
            - category_totals (dict category -> amount)
            - savings_trend (list or dict to describe trend)
            - top_merchants (list of tuples (merchant, amount))
            - anomalies_summary (text)
        """

        # header
        lines = [
            "You are a professional personal finance advisor. Analyze the user's metrics below and produce:",
            "1) A short (3-5 sentence) summary of the user's financial state.",
            "2) 5 clear, prioritized recommendations the user can implement this month (bulleted).",
            "3) Any warnings / concerning patterns (if any).",
            "4) A 1-paragraph actionable budget plan with target savings for next month.",
            "5) One-sentence habit suggestion to improve finances.",
            "",
            "Use plain language, avoid jargon, and keep suggested numeric targets realistic.",
            "",
            "User metrics:"
        ]

        # Basic numbers
        monthly_spending = metrics.get("monthly_spending")
        inflow = metrics.get("inflow")
        savings = metrics.get("savings")
        if monthly_spending is not None:
            lines.append(f"- Monthly spending: {self._format_currency(monthly_spending)}")
        if inflow is not None:
            lines.append(f"- Inflow (income): {self._format_currency(inflow)}")
        if savings is not None:
            lines.append(f"- Savings (inflow - outflow): {self._format_currency(savings)}")

        # Highest category
        highest = metrics.get("highest_category")
        if highest:
            cat, amt = highest
            lines.append(f"- Highest spending category: {cat} ({self._format_currency(amt)})")

        # Category totals (top 5)
        cat_totals = metrics.get("category_totals", {})
        if cat_totals:
            lines.append("- Category totals (top categories):")
            # sort descending
            items = sorted(cat_totals.items(), key=lambda x: -abs(x[1]))[:8]
            for c, a in items:
                lines.append(f"  - {c}: {self._format_currency(a)}")

        # Top merchants
        merchants = metrics.get("top_merchants") or []
        if merchants:
            lines.append("- Top merchants / recipients:")
            for m, a in (merchants[:5]):
                lines.append(f"  - {m}: {self._format_currency(a)}")

        # Unusual spends
        unusual = metrics.get("unusual_spends") or []
        if unusual:
            lines.append("- Unusual spends detected:")
            for u in unusual[:6]:
                m = u.get("merchant") or u.get("description") or "Unknown"
                a = u.get("amount", 0)
                note = u.get("note", "")
                lines.append(f"  - {m}: {self._format_currency(a)} {('- ' + note) if note else ''}")

        # Recurring bills
        recurring = metrics.get("recurring_bills") or []
        if recurring:
            lines.append("- Recurring monthly bills: " + ", ".join(recurring))

        # Anomalies summary
        anomalies = metrics.get("anomalies_summary")
        if anomalies:
            lines.append(f"- Anomalies summary: {anomalies}")

        # Additional context
        extra = metrics.get("extra_context")
        if extra:
            lines.append("")
            lines.append("Extra context:")
            lines.append(extra)

        # Instruction to LLM: keep output structured JSON for easier parsing
        lines.append("")
        lines.append(
            "Produce the response in JSON with keys: summary, recommendations (list), warnings (list), budget_plan, habit_tip. "
            "Also include a short 'explainers' field with 2 one-line explanations of how the calculations were made."
        )

        return "\n".join(lines)

    # ---------------------------
    # LLM CALLS (Groq)
    # ---------------------------
    def _call_groq_chat(self, prompt_text: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """
        Calls Groq chat completions API.
        Returns the assistant text.
        """
        if self.groq_client is None:
            raise RuntimeError("Groq client not initialized.")

        # Build chat messages
        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]

        try:
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.groq_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract assistant message
            content = chat_completion.choices[0].message.content
            return content

        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {str(e)}")

    # ---------------------------
    # Public API
    # ---------------------------
    def get_advice(
        self,
        metrics: Dict[str, Any],
        backend_callable=None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Build prompt and query LLM. Returns parsed JSON if possible; otherwise returns raw text under 'raw'.
        - metrics: dict (see build_prompt)
        - backend_callable: optional, if provided used to call LLM: backend_callable(prompt_text) -> str
        """

        prompt_text = self.build_prompt(metrics)

        # Prefer explicit backend_callable if provided (for local LLMs or custom wrappers)
        raw_response = None
        if backend_callable is not None:
            raw_response = backend_callable(prompt_text)
        elif self.backend == "groq":
            raw_response = self._call_groq_chat(prompt_text, temperature=temperature, max_tokens=max_tokens)
        else:
            raise RuntimeError("No valid LLM backend provided. Either set backend='groq' or pass backend_callable.")

        # Try parse JSON from model (best-effort)
        parsed = None
        try:
            # The model is asked to produce JSON — try to find a JSON substring
            start = raw_response.find("{")
            if start != -1:
                json_str = raw_response[start:]
                # Find the matching closing brace
                brace_count = 0
                end = start
                for i, char in enumerate(json_str):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = start + i + 1
                            break
                json_str = raw_response[start:end]
                parsed = json.loads(json_str)
            else:
                parsed = None
        except Exception:
            parsed = None

        result = {"raw": raw_response, "parsed": parsed}
        return result


# Module-level helper to produce advice from a prompt and optional dataframe
def answer(prompt: str, df=None, backend_callable=None):
    """Lightweight wrapper that builds simple metrics from `df` and calls LLMAdvisor.get_advice.

    If Groq is not configured, and no backend_callable provided, returns a heuristic textual response.
    """
    # Try to import pandas locally (should be available in env)
    try:
        import pandas as pd
    except Exception:
        pd = None

    # Build simple metrics if df provided
    metrics = {}
    if df is not None and pd is not None:
        try:
            df2 = df.copy()
            
            # Compute Amount column safely
            if 'Amount' in df2.columns:
                df2['Amount'] = pd.to_numeric(df2['Amount'], errors='coerce').fillna(0)
            elif 'Credit' in df2.columns and 'Debit' in df2.columns:
                df2['Credit'] = pd.to_numeric(df2['Credit'], errors='coerce').fillna(0)
                df2['Debit'] = pd.to_numeric(df2['Debit'], errors='coerce').fillna(0)
                df2['Amount'] = df2['Credit'] - df2['Debit']
            elif 'Credit' in df2.columns:
                df2['Amount'] = pd.to_numeric(df2['Credit'], errors='coerce').fillna(0)
            elif 'Debit' in df2.columns:
                df2['Amount'] = -pd.to_numeric(df2['Debit'], errors='coerce').fillna(0)
            else:
                df2['Amount'] = 0.0
            
            # Calculate metrics
            total_spending = df2.loc[df2['Amount'] < 0, 'Amount'].sum()
            total_inflow = df2.loc[df2['Amount'] > 0, 'Amount'].sum()
            net_savings = total_inflow + total_spending  # spending is negative
            
            # Top merchants by absolute spending
            top_merchants = []
            if 'Details' in df2.columns:
                top = df2[df2['Amount'] < 0].groupby('Details')['Amount'].sum().sort_values().head(10)  # most negative first
                top_merchants = [(m, abs(a)) for m, a in zip(top.index.tolist(), top.values.tolist())]
            
            # Compute unusual/anomalies
            unusual = []
            if len(df2) > 0:
                amounts = df2[df2['Amount'] < 0]['Amount'].abs()
                if len(amounts) > 0:
                    mean_amt = amounts.mean()
                    std_amt = amounts.std()
                    if std_amt > 0:
                        threshold = mean_amt + (2 * std_amt)
                        unusual_rows = df2[(df2['Amount'] < 0) & (df2['Amount'].abs() > threshold)]
                        for idx, row in unusual_rows.iterrows():
                            detail = row.get('Details', 'Unknown')
                            amt = abs(row['Amount'])
                            unusual.append({'merchant': detail, 'amount': float(amt), 'note': 'Anomaly detected'})
            
            metrics = {
                'monthly_spending': float(abs(total_spending)),
                'inflow': float(total_inflow),
                'savings': float(net_savings),
                'top_merchants': top_merchants,
                'unusual_spends': unusual[:5],
                'category_totals': {},
                'extra_context': prompt
            }
        except Exception as e:
            metrics = {'extra_context': prompt, 'note': f'Error computing metrics: {str(e)}'}

    # If no backend provided or Groq unavailable, return heuristic reply
    try:
        advisor = LLMAdvisor(backend='groq')
        return advisor.get_advice(metrics, backend_callable=backend_callable)
    except Exception as e:
        # fallback: if Groq not available, provide simple text summary
        if metrics and 'monthly_spending' in metrics:
            top_m = metrics.get('top_merchants', [])
            top_str = '; '.join([f"{m}: ₹{a:,.0f}" for m, a in top_m[:3]])
            fallback_text = (
                f"Based on your statement: You spent ₹{metrics['monthly_spending']:,.0f} this month. "
                f"Your inflow was ₹{metrics['inflow']:,.0f}. Top spending: {top_str}. "
                f"To get AI-powered recommendations, set up GROQ_API_KEY environment variable."
            )
        else:
            fallback_text = f"Heuristic reply: Could not analyze statement data. Error: {str(e)}"
        
        return {
            'raw': fallback_text,
            'parsed': None
        }


# Example usage
if __name__ == "__main__":
    # Initialize advisor with Groq
    advisor = LLMAdvisor(
        backend="groq",
        groq_model="llama-3.3-70b-versatile",  # Fast and capable model
        groq_api_key=os.getenv("GROQ_API_KEY")  # Or pass directly
    )
    
    # Sample metrics
    sample_metrics = {
        "monthly_spending": 45000,
        "inflow": 60000,
        "savings": 15000,
        "highest_category": ("Food & Dining", 12000),
        "category_totals": {
            "Food & Dining": 12000,
            "Transportation": 8000,
            "Shopping": 7000,
            "Entertainment": 5000,
            "Utilities": 4000
        },
        "top_merchants": [
            ("Swiggy", 4500),
            ("Uber", 3200),
            ("Amazon", 6000)
        ],
        "unusual_spends": [
            {"merchant": "Electronics Store", "amount": 15000, "note": "One-time purchase"}
        ],
        "recurring_bills": ["Netflix ₹649", "Spotify ₹119", "Internet ₹999"]
    }
    
    # Get advice
    result = advisor.get_advice(sample_metrics)
    
    print("Raw Response:")
    print(result["raw"])
    print("\n" + "="*50 + "\n")
    
    if result["parsed"]:
        print("Parsed Response:")
        print(json.dumps(result["parsed"], indent=2))