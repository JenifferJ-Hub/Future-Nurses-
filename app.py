from __future__ import annotations

import json
import math
import uuid
import hashlib
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:  # allows the deterministic engine tests to run before UI dependencies are installed
    st = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "sample_data"
SNAPSHOT_DIR = APP_DIR / "snapshots"
EXPECTED_TEAM = "Future Nurses"
EXPECTED_VERSION = 1
CURRENCY = "USD"

DEFAULTS = {
    "team_id": EXPECTED_TEAM,
    "course": "Precalculus",
    "pathway_id": "NUR-2",
    "business_name": "Nurse-led health education simulation",
    "selected_level": "Advanced",
    "snapshot_id": "future-nurses-stage1",
    "state_version": EXPECTED_VERSION,
    "simulation_date": "2027-01-01",
    "period_unit": "Monthly",
    "random_seed": 42,
    "working_days": 20,
    "base_price": "25.00",
    "D0": 80,
    "staff_hours": "30.0",
    "hours_per_unit": "0.5",
    "venue_seats": 60,
    "no_show_rate": "0.0",
    "satisfaction_mean": "4.0",
    "owner_contribution": "6000.00",
    "equipment_assets": "1500.00",
    "preopening_expense": "500.00",
    "opening_inventory": "0.00",
    "opening_cash": "4000.00",
    "opening_retained_earnings": "-500.00",
    "variable_costs": {
        "educator_delivery": "2.00",
        "materials": "2.00",
        "cleaning_consumables": "1.00",
    },
    "fixed_costs": {
        "venue": "250.00",
        "scheduling_software": "100.00",
        "accessibility_support": "200.00",
        "admin_insurance_assumption": "150.00",
        "advertising_campaign": "0.00",
    },
    "refund_amount": "0.00",
    "loan_principal": "0.00",
    "interest": "0.00",
    "additional_capex": "0.00",
    "new_owner_cash": "0.00",
    "principal_repayment": "0.00",
    "enable_depreciation": False,
    "depreciation": "0.00",
    "teacher_approval": False,
}

OFFERINGS = [
    {"offering_id": "general_workshop", "name": "General wellness education workshop",
     "status": "Primary", "allocation_rule": "Uses the main monthly capacity and cost model; only offering counted in Stage 1 ledger."},
    {"offering_id": "caregiver_session", "name": "Caregiver information session",
     "status": "Secondary placeholder", "allocation_rule": "Separate capacity/cost allocation required before activation; excluded from Stage 1 revenue."},
    {"offering_id": "first_aid_demo", "name": "First-aid awareness demonstration",
     "status": "Secondary placeholder", "allocation_rule": "Separate capacity/cost allocation required before activation; excluded from Stage 1 revenue."},
]

REQUIRED_SNAPSHOT = {
    "team_id", "course", "pathway_id", "snapshot_id", "state_version",
    "simulation_date", "base_period", "working_days", "currency",
    "level_by_skill", "assumptions", "sources", "offerings", "funding",
    "cost_lines", "capacity", "ledger", "campaigns", "decisions",
}
LEDGER_FIELDS = {
    "transaction_id", "date", "event_type", "category", "amount", "quantity",
    "unit_price", "offering_id", "campaign_id", "financing_type",
    "state_version", "run_id",
}

def D(value: Any) -> Decimal:
    return Decimal(str(value))

def cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def money(value: Any) -> str:
    return f"${cents(D(value)):,.2f}"

def iso_date(value: Any) -> date:
    return date.fromisoformat(str(value))

def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def deterministic_run_id(inputs: dict, mode: str = "normal") -> str:
    # Same mode + same simulation inputs = same run ID, preventing accidental duplicate reruns.
    canonical = json.dumps({"mode": mode, "inputs": inputs}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{mode}-{digest}"

def calculate_month(requested: int, capacity: int, price: Decimal,
                    variable_unit_cost: Decimal, fixed_cost: Decimal,
                    marketing: Decimal, no_show_rate: Decimal,
                    opening_cash: Decimal, interest: Decimal = Decimal("0"),
                    principal: Decimal = Decimal("0"), capex: Decimal = Decimal("0"),
                    new_owner_cash: Decimal = Decimal("0"), loan_cash: Decimal = Decimal("0"),
                    refund: Decimal = Decimal("0"), depreciation: Decimal = Decimal("0")) -> dict:
    booked = min(int(requested), int(capacity))
    attended = math.floor(booked * (Decimal("1") - no_show_rate))
    sold = attended
    revenue_gross = cents(price * sold)
    refunds = min(cents(refund), revenue_gross)
    revenue = cents(revenue_gross - refunds)
    variable_expense = cents(variable_unit_cost * sold)
    operating_profit = cents(revenue - variable_expense - fixed_cost - marketing)
    net_profit = cents(operating_profit - interest - depreciation)
    ending_cash = cents(
        opening_cash + revenue - variable_expense - fixed_cost - marketing
        - interest - principal - capex + new_owner_cash + loan_cash
    )
    return {
        "booked": booked, "attended": attended, "sold": sold,
        "unserved_demand": max(0, int(requested) - booked),
        "revenue": revenue, "refunds": refunds,
        "variable_expense": variable_expense, "fixed_expense": cents(fixed_cost),
        "marketing": cents(marketing), "operating_profit": operating_profit,
        "net_profit": net_profit, "ending_cash": ending_cash,
    }

def make_entry(event_date: str, event_type: str, category: str, amount: Decimal,
               quantity: int = 0, unit_price: Decimal = Decimal("0"),
               offering_id: str = "general_workshop", campaign_id: str | None = None,
               financing_type: str | None = None, run_id: str = "") -> dict:
    return {
        "transaction_id": make_id("txn"), "date": event_date,
        "event_type": event_type, "category": category,
        "amount": str(cents(amount)), "quantity": int(quantity),
        "unit_price": str(cents(unit_price)), "offering_id": offering_id,
        "campaign_id": campaign_id, "financing_type": financing_type,
        "state_version": EXPECTED_VERSION, "run_id": run_id,
    }

def build_monthly_run(inputs: dict, run_id: str, capacity_override: int | None = None,
                      fixture_mode: bool = False) -> tuple[list[dict], dict]:
    sim_date = iso_date(inputs["simulation_date"])
    month_date = sim_date.replace(day=1).isoformat()
    variable_total = sum((D(v) for v in inputs["variable_costs"].values()), Decimal("0"))
    fixed_total = sum((D(v) for k, v in inputs["fixed_costs"].items() if k != "advertising_campaign"), Decimal("0"))
    marketing = D(inputs["fixed_costs"]["advertising_campaign"])
    if fixture_mode:
        capacity = int(capacity_override)
        requested = int(inputs["D0"])
    else:
        educator_capacity = math.floor(D(inputs["staff_hours"]) / D(inputs["hours_per_unit"]))
        capacity = min(educator_capacity, int(inputs["venue_seats"]))
        requested = int(inputs["D0"])
    result = calculate_month(
        requested, capacity, D(inputs["base_price"]), variable_total, fixed_total,
        marketing, D(inputs["no_show_rate"]), D(inputs["opening_cash"]),
        interest=D(inputs["interest"]), principal=D(inputs["principal_repayment"]),
        capex=D(inputs["additional_capex"]), new_owner_cash=D(inputs["new_owner_cash"]),
        loan_cash=D(inputs["loan_principal"]), refund=D(inputs["refund_amount"]),
        depreciation=D(inputs["depreciation"]) if inputs["enable_depreciation"] else Decimal("0"),
    )
    entries = []
    entries.append(make_entry(month_date, "sale", "workshop_revenue", result["revenue"],
                              result["sold"], D(inputs["base_price"]), run_id=run_id))
    entries.append(make_entry(month_date, "expense", "variable_delivery_cost", result["variable_expense"],
                              result["sold"], variable_total, run_id=run_id))
    entries.append(make_entry(month_date, "expense", "fixed_operating_cost", fixed_total, run_id=run_id))
    if marketing:
        entries.append(make_entry(month_date, "expense", "advertising", marketing, run_id=run_id))
    if result["refunds"]:
        entries.append(make_entry(month_date, "refund", "customer_refund", result["refunds"], run_id=run_id))
    if D(inputs["interest"]):
        entries.append(make_entry(month_date, "expense", "loan_interest", D(inputs["interest"]), run_id=run_id))
    if D(inputs["principal_repayment"]):
        entries.append(make_entry(month_date, "financing", "principal_repayment", D(inputs["principal_repayment"]),
                                  financing_type="principal", run_id=run_id))
    if D(inputs["loan_principal"]):
        entries.append(make_entry(month_date, "financing", "loan_proceeds", D(inputs["loan_principal"]),
                                  financing_type="loan", run_id=run_id))
    if D(inputs["new_owner_cash"]):
        entries.append(make_entry(month_date, "financing", "owner_contribution", D(inputs["new_owner_cash"]),
                                  financing_type="owner_equity", run_id=run_id))
    if D(inputs["additional_capex"]):
        entries.append(make_entry(month_date, "asset", "capital_expenditure", D(inputs["additional_capex"]), run_id=run_id))
    if result["refunds"]:
        # Refund is represented as a reduction to sales revenue in the summary and as a separate cash event.
        pass
    summary = {**result, "capacity": capacity, "requested": requested,
               "variable_unit_cost": variable_total, "fixed_cost_total": fixed_total,
               "run_id": run_id, "month": month_date, "fixture_mode": fixture_mode}
    return entries, summary

def snapshot_from_state(inputs: dict, ledger: list[dict], run_id: str | None = None) -> dict:
    return {
        "team_id": EXPECTED_TEAM, "course": "Precalculus", "pathway_id": "NUR-2",
        "snapshot_id": inputs["snapshot_id"], "state_version": EXPECTED_VERSION,
        "simulation_date": inputs["simulation_date"], "base_period": "month",
        "working_days": int(inputs["working_days"]), "currency": CURRENCY,
        "level_by_skill": {"Rational Average Cost": "Foundation",
                           "Function Composition & Constraints": "Applied",
                           "Seasonal Trigonometric Model": inputs["selected_level"]},
        "assumptions": inputs, "sources": ["Fictional editable classroom defaults supplied in project brief"],
        "offerings": OFFERINGS,
        "funding": {"owner_contribution": inputs["owner_contribution"],
                    "loan_principal": inputs["loan_principal"],
                    "new_owner_cash": inputs["new_owner_cash"]},
        "cost_lines": {"variable": inputs["variable_costs"], "fixed": inputs["fixed_costs"]},
        "capacity": {"staff_hours": inputs["staff_hours"], "hours_per_unit": inputs["hours_per_unit"],
                     "venue_seats": inputs["venue_seats"]},
        "ledger": ledger, "campaigns": [], "decisions": [],
        "last_run_id": run_id,
    }

def validate_snapshot(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Snapshot must be a JSON object.")
    missing = REQUIRED_SNAPSHOT - set(data)
    if missing:
        raise ValueError("Snapshot is missing required fields: " + ", ".join(sorted(missing)))
    if data["team_id"] != EXPECTED_TEAM:
        raise ValueError(f"Wrong team: expected '{EXPECTED_TEAM}', received '{data['team_id']}'.")
    if type(data["state_version"]) is not int or data["state_version"] != EXPECTED_VERSION:
        raise ValueError(f"Stale or unsupported state_version. Expected exactly {EXPECTED_VERSION}.")
    try:
        date.fromisoformat(data["simulation_date"])
    except Exception as exc:
        raise ValueError("simulation_date must be a valid ISO date (YYYY-MM-DD).") from exc
    if not isinstance(data["ledger"], list):
        raise ValueError("ledger must be a list.")
    seen = set()
    for i, entry in enumerate(data["ledger"]):
        if not isinstance(entry, dict) or not LEDGER_FIELDS.issubset(entry):
            raise ValueError(f"Ledger entry {i + 1} is missing required fields.")
        if entry["transaction_id"] in seen:
            raise ValueError(f"Duplicate transaction_id: {entry['transaction_id']}")
        seen.add(entry["transaction_id"])
        try:
            date.fromisoformat(entry["date"])
            if int(entry["quantity"]) < 0:
                raise ValueError("quantity cannot be negative")
            Decimal(str(entry["amount"]))
            Decimal(str(entry["unit_price"]))
        except Exception as exc:
            raise ValueError(f"Invalid ledger entry {i + 1}: {exc}") from exc
        if type(entry["state_version"]) is not int or entry["state_version"] != EXPECTED_VERSION:
            raise ValueError(f"Ledger entry {i + 1} has an unsupported state_version.")
        if not isinstance(entry["run_id"], str) or not entry["run_id"]:
            raise ValueError(f"Ledger entry {i + 1} needs a run_id.")

def aggregate_ledger(ledger: list[dict], frequency: str) -> pd.DataFrame:
    if not ledger:
        return pd.DataFrame(columns=["period", "event_count", "net_amount", "quantity"])
    df = pd.DataFrame(ledger)
    df["date"] = pd.to_datetime(df["date"])
    df["amount_num"] = pd.to_numeric(df["amount"])
    if frequency == "Daily":
        df["period"] = df["date"].dt.strftime("%Y-%m-%d")
    elif frequency == "Yearly":
        df["period"] = df["date"].dt.strftime("%Y")
    else:
        df["period"] = df["date"].dt.strftime("%Y-%m")
    return (df.groupby("period", as_index=False)
            .agg(event_count=("transaction_id", "count"),
                 net_amount=("amount_num", "sum"),
                 quantity=("quantity", "sum")))

def get_inputs_from_session() -> dict:
    return {
        "team_id": EXPECTED_TEAM, "course": "Precalculus", "pathway_id": "NUR-2",
        "business_name": st.session_state.business_name,
        "selected_level": st.session_state.selected_level,
        "snapshot_id": st.session_state.snapshot_id,
        "simulation_date": st.session_state.simulation_date.isoformat(),
        "period_unit": st.session_state.period_unit,
        "random_seed": int(st.session_state.random_seed),
        "working_days": int(st.session_state.working_days),
        "base_price": str(st.session_state.base_price), "D0": int(st.session_state.D0),
        "staff_hours": str(st.session_state.staff_hours),
        "hours_per_unit": str(st.session_state.hours_per_unit),
        "venue_seats": int(st.session_state.venue_seats),
        "no_show_rate": str(st.session_state.no_show_rate),
        "satisfaction_mean": str(st.session_state.satisfaction_mean),
        "owner_contribution": str(st.session_state.owner_contribution),
        "equipment_assets": str(st.session_state.equipment_assets),
        "preopening_expense": str(st.session_state.preopening_expense),
        "opening_inventory": str(st.session_state.opening_inventory),
        "opening_cash": str(st.session_state.opening_cash),
        "opening_retained_earnings": str(st.session_state.opening_retained_earnings),
        "variable_costs": {k: str(st.session_state[f"var_{k}"]) for k in DEFAULTS["variable_costs"]},
        "fixed_costs": {k: str(st.session_state[f"fixed_{k}"]) for k in DEFAULTS["fixed_costs"]},
        "refund_amount": str(st.session_state.refund_amount),
        "loan_principal": str(st.session_state.loan_principal),
        "interest": str(st.session_state.interest),
        "additional_capex": str(st.session_state.additional_capex),
        "new_owner_cash": str(st.session_state.new_owner_cash),
        "principal_repayment": str(st.session_state.principal_repayment),
        "enable_depreciation": bool(st.session_state.enable_depreciation),
        "depreciation": str(st.session_state.depreciation),
        "teacher_approval": bool(st.session_state.teacher_approval),
    }

def validate_inputs(x: dict) -> list[str]:
    errors = []
    try:
        if D(x["base_price"]) <= 0: errors.append("Base price must be greater than zero.")
        if int(x["D0"]) < 0: errors.append("D0 must be a nonnegative integer.")
        if D(x["staff_hours"]) < 0: errors.append("Staff hours cannot be negative.")
        if D(x["hours_per_unit"]) <= 0: errors.append("Hours per unit must be greater than zero.")
        if int(x["venue_seats"]) < 0: errors.append("Venue seats cannot be negative.")
        if not (Decimal("0") <= D(x["no_show_rate"]) <= Decimal("1")): errors.append("No-show rate must be between 0 and 1.")
        if not (Decimal("1") <= D(x["satisfaction_mean"]) <= Decimal("5")): errors.append("Satisfaction mean must be between 1 and 5.")
        for group in (x["variable_costs"], x["fixed_costs"]):
            for k, v in group.items():
                if D(v) < 0: errors.append(f"{k} cannot be negative.")
        for k in ("refund_amount", "loan_principal", "interest", "additional_capex", "new_owner_cash", "principal_repayment", "depreciation"):
            if D(x[k]) < 0: errors.append(f"{k} cannot be negative.")
    except Exception as exc:
        errors.append(f"Invalid numeric input: {exc}")
    return errors

def render_math_panel(inputs: dict):
    st.header("Math Skills & Challenge Levels")
    st.caption("Course: Precalculus • Three level pathways are available. Teacher approval is required before a level change is treated as approved.")
    levels = ["Foundation", "Applied", "Advanced"]
    current = st.selectbox("Selected challenge level", levels, index=levels.index(inputs["selected_level"]))
    approved = st.toggle("Teacher approval recorded for selected level", value=inputs["teacher_approval"])
    st.session_state.selected_level = current
    st.session_state.teacher_approval = approved
    if current != "Advanced" and not approved:
        st.info("Level selection is saved as a student proposal until teacher approval is enabled.")
    tab1, tab2, tab3 = st.tabs(["Foundation · Average cost", "Applied · Composition", "Advanced · Seasonality"])

    with tab1:
        st.subheader("Rational Average Cost")
        st.write("**Business question:** How does average cost per attended workshop seat change as monthly sales increase?")
        st.latex(r"C(s)=F+v\\cdot s,\\qquad A(s)=\\frac{F}{s}+v")
        st.write("Variables: s = attended seats (seats/month); F = $700/month fixed operating cost; v = $5/seat variable cost. Domain: 0 < s ≤ 60 seats/month. Assumes baseline costs and no advertising.")
        xs = list(range(1, 61))
        ys = [700 / s + 5 for s in xs]
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label="A(s)")
        ax.axhline(5, linestyle="--", label="Horizontal asymptote y = 5")
        ax.axvline(0, linestyle=":", label="Vertical asymptote s = 0")
        ax.set(xlabel="Attended seats per month (s)", ylabel="Average cost ($/seat)", title="Rational average-cost curve")
        ax.set_xlim(0, 62); ax.set_ylim(0, max(ys) * 1.05); ax.legend()
        st.pyplot(fig)
        table = pd.DataFrame({"s (seats)": xs, "C(s) total cost ($)": [700 + 5*s for s in xs],
                              "A(s) average cost ($/seat)": ys})
        chosen = st.selectbox("Highlight a data-table row (seats)", xs, index= min(9, len(xs)-1), key="foundation_row")
        st.dataframe(table.style.apply(lambda r: ["background-color: #fff2cc" if r["s (seats)"] == chosen else "" for _ in r], axis=1), use_container_width=True)
        st.caption("A(0) = Undefined / N/A (division by zero). The horizontal asymptote y = 5 is a mathematical limit; practical capacity is at most 60 seats.")
        st.download_button("Download chart (PNG)", fig_to_bytes(fig, "png"), "average_cost.png", "image/png")
        st.download_button("Download chart (SVG)", fig_to_bytes(fig, "svg"), "average_cost.svg", "image/svg+xml")
        st.download_button("Export data + assumptions (CSV)", table.to_csv(index=False).encode(), "average_cost.csv", "text/csv")
        st.text_input("Interpretation prompt", value="What happens to average cost as more seats share the fixed cost?", disabled=True)
        plt.close(fig)

    with tab2:
        st.subheader("Function Composition & Constraints")
        st.write("**Business question:** How does changing the seat price affect demand, realized sales, revenue, cost, and profit?")
        st.latex(r"q(p)=\\max(0,160-3.2p),\\quad s(p)=\\min(60,q(p))")
        st.latex(r"R(p)=p\\,s(p),\\quad C(s(p))=700+5s(p),\\quad P(p)=R(p)-C(s(p))")
        st.write("p = price (USD/seat), q = requested seats/month, s = realized seats/month. Domain: 0 ≤ p ≤ 50 USD. Capacity = 60 seats/month. Continuous projections are shown separately from integer transaction rounding.")
        prices = [i / 2 for i in range(0, 101)]
        rows = []
        for p in prices:
            q = max(0, 160 - 3.2*p)
            s = min(60, q)
            rev = p*s; cost = 700 + 5*s
            rows.append({"Price ($)": p, "Demand (seats)": q, "Realized sales (continuous)": s,
                         "Revenue ($)": rev, "Cost ($)": cost, "Profit ($)": rev-cost,
                         "Rounded sales (seats)": math.floor(s)})
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots()
        ax.plot(df["Price ($)"], df["Profit ($)"], label="Continuous projected profit")
        ax.axhline(0, linestyle="--")
        ax.set(xlabel="Price ($/seat)", ylabel="Profit ($/month)", title="Composed profit function")
        ax.legend()
        st.pyplot(fig)
        bp = 25 * (2 - 60/80) if 80 else None
        st.metric("Capacity breakpoint price", f"${bp:.2f}" if bp is not None and 0 <= bp <= 50 else "Outside domain")
        st.caption("Breakpoint formula: p = p₀ × (2 − capacity/D₀), when D₀ > 0 and the result is in [0, 50].")
        selected_price = st.select_slider("Highlight price row ($)", options=[float(x) for x in prices], value=25.0)
        st.dataframe(df.style.apply(lambda r: ["background-color: #fff2cc" if r["Price ($)"] == selected_price else "" for _ in r], axis=1), use_container_width=True)
        st.download_button("Download chart (PNG)", fig_to_bytes(fig, "png"), "composed_profit.png", "image/png")
        st.download_button("Download chart (SVG)", fig_to_bytes(fig, "svg"), "composed_profit.svg", "image/svg+xml")
        st.download_button("Export mapping table (CSV)", df.to_csv(index=False).encode(), "price_mapping.csv", "text/csv")
        st.info("Interpretation prompt: At what price does capacity stop constraining demand, and how does rounding change the number of seats actually sold?")
        plt.close(fig)

    with tab3:
        st.subheader("Seasonal Trigonometric Model")
        st.write("**Business question:** How might monthly interest in a general education workshop vary seasonally, before capacity limits?")
        st.latex(r"D(t)=D_0\\left(1+0.20\\sin\\left(\\frac{2\\pi(t-1)}{12}\\right)\\right)")
        st.write("t = month index (1–12); D₀ = 80 requested seats/month; period = 12 months; amplitude = 16 seats. Peak at t = 4 (96 seats); minimum at t = 10 (64 seats). Capacity clips realized sales at 60 seats.")
        months = list(range(1, 13))
        seasonal = [80*(1+0.20*math.sin(2*math.pi*(t-1)/12)) for t in months]
        cap = min(math.floor(D(inputs["staff_hours"])/D(inputs["hours_per_unit"])), int(inputs["venue_seats"]))
        scenarios = {"Baseline 1.00": 1.00, "Upside 1.10": 1.10, "Downside 0.90": 0.90}
        fig, ax = plt.subplots()
        seasonal_df = pd.DataFrame({"Month index": months, "Unclipped demand": seasonal,
                                    "Realized sales (capacity clipped)": [min(cap, x) for x in seasonal]})
        ax.plot(months, seasonal_df["Unclipped demand"], marker="o", label="Unclipped demand")
        ax.plot(months, seasonal_df["Realized sales (capacity clipped)"], marker="s", label="Realized sales")
        ax.set(xlabel="Month index (1–12)", ylabel="Seats/month", title="Seasonal demand vs. capacity")
        ax.set_xticks(months); ax.legend()
        st.pyplot(fig)
        scenario_rows = []
        for label, factor in scenarios.items():
            for t, demand in zip(months, seasonal):
                scenario_rows.append({"Scenario": label, "Month": t, "Unclipped demand": demand*factor,
                                      "Realized sales": min(cap, demand*factor)})
        scenario_df = pd.DataFrame(scenario_rows)
        st.dataframe(seasonal_df, use_container_width=True)
        st.write("Scenario comparison (kept separate from the actual dated ledger)")
        st.dataframe(scenario_df, use_container_width=True)
        st.download_button("Download chart (PNG)", fig_to_bytes(fig, "png"), "seasonal_demand.png", "image/png")
        st.download_button("Download chart (SVG)", fig_to_bytes(fig, "svg"), "seasonal_demand.svg", "image/svg+xml")
        st.download_button("Export seasonal data (CSV)", scenario_df.to_csv(index=False).encode(), "seasonal_scenarios.csv", "text/csv")
        st.info("Interpretation prompt: Which months exceed capacity, and why is the scenario table not recorded as actual sales?")
        st.caption("Assumptions: fictional planning model; fixed 20% seasonal amplitude; one-year cycle only. Do not extrapolate this pattern as a forecast of health outcomes or clinical need.")
        plt.close(fig)

def fig_to_bytes(fig, fmt: str) -> bytes:
    import io
    out = io.BytesIO()
    fig.savefig(out, format=fmt, bbox_inches="tight")
    return out.getvalue()

def main():
    st.set_page_config(page_title="Career Venture Lab · Future Nurses", layout="wide")
    st.title("Career Venture Lab")
    st.caption("Stage 1 · Precalculus | Robotics-style deterministic business simulation framework adapted for Future Nurses")
    if "ledger" not in st.session_state: st.session_state.ledger = []
    if "run_ids" not in st.session_state: st.session_state.run_ids = set()
    if "last_summary" not in st.session_state: st.session_state.last_summary = None
    if "snapshot_id" not in st.session_state: st.session_state.snapshot_id = DEFAULTS["snapshot_id"]

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Stage 1 workspace", ["Overview & System Map", "Inputs & Startup Financing",
            "Math Skills & Challenge Levels", "Simulation & Event Ledger", "Save / Load & Exports",
            "Stage 2 · Department Handoffs (placeholder)", "Stage 3 · Assessments (placeholder)",
            "Stage 4 · Optional Storefront (placeholder)"])
        st.divider()
        st.caption("Team: Future Nurses")
        st.caption("Course: Precalculus · Pathway: NUR-2")
        st.caption("State version: 1")

    # Create all editable widgets once, retaining Streamlit's normal rerun behavior.
    if "widgets_initialized" not in st.session_state:
        for k, v in DEFAULTS.items():
            if k in ("variable_costs", "fixed_costs"): continue
            if k == "simulation_date": st.session_state[k] = date.fromisoformat(v)
            elif k not in st.session_state:
                if k in {"base_price", "staff_hours", "hours_per_unit", "no_show_rate",
                         "satisfaction_mean", "owner_contribution", "equipment_assets",
                         "preopening_expense", "opening_inventory", "opening_cash",
                         "opening_retained_earnings", "refund_amount", "loan_principal",
                         "interest", "additional_capex", "new_owner_cash",
                         "principal_repayment", "depreciation"}:
                    st.session_state[k] = float(v)
                else:
                    st.session_state[k] = v
        for k, v in DEFAULTS["variable_costs"].items():
            if f"var_{k}" not in st.session_state: st.session_state[f"var_{k}"] = float(v)
        for k, v in DEFAULTS["fixed_costs"].items():
            if f"fixed_{k}" not in st.session_state: st.session_state[f"fixed_{k}"] = float(v)
        st.session_state.widgets_initialized = True

    if page == "Overview & System Map":
        st.header("Overview & System Map")
        st.write("**Business concept:** fictional, nonclinical general health-education workshops. No diagnosis, clinical care, or patient data are modeled.")
        st.markdown("""
        ```text
        Campaign → Demand → Capacity → Delivery → Satisfaction → Later demand
           │          │         │          │            │
        Student-      Automatic Automatic  Automatic    Student-reviewed
        managed       model     model      model        survey summary only
        ```
        """)
        st.markdown("""
        - **Student-managed:** campaign budget, price, opening assumptions, scheduling choices, interpretation, and review of supplied satisfaction summaries.
        - **Automatic:** deterministic demand/capacity calculation, bookings, attendance, transactions, costs, profit, cash roll-forward, and ledger aggregation.
        - **Feedback loop guardrail:** satisfaction is displayed as a supplied survey summary; it does not infer health outcomes or automatically increase demand.
        """)
        st.subheader("Department dependencies")
        deps = pd.DataFrame([
            ["Marketing / outreach", "Campaign budget and message", "Demand input"],
            ["Operations / scheduling", "Staff hours, venue seats, working days", "Capacity limit"],
            ["Education delivery", "Attended seats and educator time", "Variable delivery cost"],
            ["Finance / accounting", "Funding, costs, refunds, assets, debt", "Profit and cash reports"],
            ["Quality review", "Supplied satisfaction mean", "Human review only; no automatic demand change"],
        ], columns=["Department", "Provides", "Depends on"])
        st.dataframe(deps, hide_index=True, use_container_width=True)
        st.subheader("Stage roadmap")
        st.success("Stage 1 · Overview, inputs, Precalculus models, deterministic engine, dated ledger, snapshots, exports, tests")
        st.info("Stage 2 · Department Handoffs, Full Report Center, Costed Marketing Studio — placeholder")
        st.info("Stage 3 · Assessments, Cross-Course Connections — placeholder")
        st.info("Stage 4 · Optional Storefront — placeholder")
        st.caption("No live AI responses, clinical recommendations, patient records, or automatic publishing.")

    elif page == "Inputs & Startup Financing":
        st.header("Inputs & Startup Financing")
        st.subheader("Team & control fields")
        c1, c2, c3 = st.columns(3)
        c1.text_input("Team ID", value=EXPECTED_TEAM, disabled=True)
        c2.text_input("Pathway ID", value="NUR-2", disabled=True)
        c3.text_input("Selected course", value="Precalculus", disabled=True)
        st.text_input("Business name", key="business_name")
        c1, c2, c3 = st.columns(3)
        c1.date_input("Simulation date", key="simulation_date")
        c2.selectbox("Period unit", ["Daily", "Monthly", "Yearly"], key="period_unit")
        c3.number_input("Working days/month", min_value=1, max_value=31, key="working_days")
        c1, c2, c3 = st.columns(3)
        c1.text_input("Snapshot ID", key="snapshot_id")
        c2.number_input("Random seed", min_value=0, key="random_seed")
        c3.number_input("State version", value=EXPECTED_VERSION, disabled=True)

        st.subheader("Demand, price & capacity")
        a, b, c = st.columns(3)
        a.number_input("Base price ($/seat)", min_value=0.01, step=1.0, format="%.2f", key="base_price")
        b.number_input("D0 requested seats/month", min_value=0, step=1, key="D0")
        c.number_input("Staff productive hours/month", min_value=0.0, step=1.0, key="staff_hours")
        a, b, c = st.columns(3)
        a.number_input("Hours per delivered seat", min_value=0.01, step=0.1, key="hours_per_unit")
        b.number_input("Venue seats/month", min_value=0, step=1, key="venue_seats")
        c.number_input("No-show rate", min_value=0.0, max_value=1.0, step=0.01, key="no_show_rate")
        st.number_input("Supplied satisfaction mean (1–5)", min_value=1.0, max_value=5.0, step=0.1, key="satisfaction_mean")

        st.subheader("Opening balances & startup financing")
        st.caption("Accounting rule: opening assets = owner equity − pre-opening expense. Equipment and inventory are assets, not recurring expenses.")
        cols = st.columns(3)
        cols[0].number_input("Owner contribution ($)", min_value=0.0, step=100.0, key="owner_contribution")
        cols[1].number_input("Equipment assets ($)", min_value=0.0, step=100.0, key="equipment_assets")
        cols[2].number_input("Pre-opening expense ($)", min_value=0.0, step=50.0, key="preopening_expense")
        cols = st.columns(3)
        cols[0].number_input("Opening inventory ($)", min_value=0.0, step=50.0, key="opening_inventory")
        cols[1].number_input("Opening spendable cash ($)", min_value=0.0, step=100.0, key="opening_cash")
        cols[2].number_input("Opening retained earnings ($)", step=100.0, key="opening_retained_earnings")
        opening_assets = D(st.session_state.owner_contribution) + D(st.session_state.opening_retained_earnings)
        st.metric("Opening assets check (equity + retained earnings)", money(opening_assets))
        st.caption("Baseline balance sheet: $6,000 owner equity − $500 pre-opening expense = $5,500 opening assets, comprising $4,000 cash + $1,500 equipment + $0 inventory.")

        st.subheader("Variable cost lines (per attended/delivered seat)")
        for key, label in [("educator_delivery","Educator delivery"),("materials","Materials"),("cleaning_consumables","Cleaning consumables")]:
            st.number_input(f"{label} ($/seat)", min_value=0.0, step=0.25, key=f"var_{key}")
        st.subheader("Monthly fixed cost lines")
        for key, label in [("venue","Venue"),("scheduling_software","Scheduling software"),("accessibility_support","Accessibility support"),("admin_insurance_assumption","Admin / insurance assumption"),("advertising_campaign","Advertising campaign (separate operating line)")]:
            st.number_input(f"{label} ($/month)", min_value=0.0, step=10.0, key=f"fixed_{key}")

        st.subheader("Optional financing, refund & capex controls")
        st.caption("All controls are disabled at the baseline amount of $0.00; enable only for a scenario.")
        for key, label in [("refund_amount","Refund amount"),("loan_principal","Loan proceeds / principal"),("interest","Interest expense"),("additional_capex","Additional capital expenditure"),("new_owner_cash","New owner cash contribution"),("principal_repayment","Principal repayment"),("depreciation","Enabled depreciation amount")]:
            st.number_input(f"{label} ($)", min_value=0.0, step=10.0, key=key)
        st.checkbox("Enable depreciation expense", key="enable_depreciation")
        st.checkbox("Teacher approval recorded", key="teacher_approval")

        x = get_inputs_from_session()
        errors = validate_inputs(x)
        if errors:
            for e in errors: st.error(e)
        else:
            st.success("Inputs pass basic range validation.")
        st.write("**Base offerings**")
        st.dataframe(pd.DataFrame(OFFERINGS), hide_index=True, use_container_width=True)

    elif page == "Math Skills & Challenge Levels":
        render_math_panel(get_inputs_from_session())

    elif page == "Simulation & Event Ledger":
        st.header("Core Deterministic Simulation Engine")
        st.write("Normal Mode computes educator capacity from staff hours and venue seats. Only attended seats generate revenue and per-seat variable expense.")
        inputs = get_inputs_from_session()
        errors = validate_inputs(inputs)
        if errors:
            for e in errors: st.error(e)
        else:
            cap = min(math.floor(D(inputs["staff_hours"])/D(inputs["hours_per_unit"])), inputs["venue_seats"])
            st.metric("Normal-mode monthly capacity", f"{cap} seats")
            st.caption("educator_capacity = floor(staff_hours / hours_per_unit); capacity = min(educator_capacity, venue_seats).")
            if st.button("Run normal monthly simulation", type="primary"):
                run_id = deterministic_run_id(inputs, "normal")
                existing_run_ids = {entry["run_id"] for entry in st.session_state.ledger}
                if run_id not in st.session_state.run_ids and run_id not in existing_run_ids:
                    entries, summary = build_monthly_run(inputs, run_id)
                    st.session_state.ledger.extend(entries)
                    st.session_state.run_ids.add(run_id)
                    st.session_state.last_summary = summary
                    st.success(f"Run recorded: {run_id}")
            if st.session_state.last_summary:
                s = st.session_state.last_summary
                st.subheader("Latest run summary")
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Requested", s["requested"]); m2.metric("Sold / attended", s["sold"])
                m3.metric("Unserved demand", s["unserved_demand"]); m4.metric("Ending cash", money(s["ending_cash"]))
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Revenue", money(s["revenue"])); m2.metric("Operating profit", money(s["operating_profit"]))
                m3.metric("Net profit", money(s["net_profit"])); m4.metric("Capacity", s["capacity"])
                st.caption("Loans and owner contributions are financing, not revenue. Principal repayment reduces cash and debt only; interest is an expense. Capex reduces cash and creates an asset, not an operating expense.")
        st.divider()
        st.subheader("Fixture-Test Mode (isolated replay)")
        st.warning("FIXTURE-TEST MODE: explicit capacity_override is used only while replaying sample_data/monthly_fixture.csv. Do not mix fixture replay and Normal Mode in the same run.")
        fixture_file = DATA_DIR / "monthly_fixture.csv"
        if st.button("Replay supplied fixture in isolated mode"):
            fixture = pd.read_csv(fixture_file, dtype={"month": str})
            fixture_run = deterministic_run_id({"inputs": inputs, "fixture": fixture_file.read_text(encoding="utf-8")}, "fixture")
            fixture_entries = []
            fixture_summaries = []
            fixture_cash = D(inputs["opening_cash"])
            for row in fixture.to_dict("records"):
                fi = dict(inputs)
                fi["opening_cash"] = str(fixture_cash)
                fi["simulation_date"] = f"{row['month']}-01"
                fi["D0"] = int(row["requested"])
                fi["fixed_costs"] = dict(inputs["fixed_costs"])
                fi["fixed_costs"]["advertising_campaign"] = str(row["ad_cost"])
                ent, summ = build_monthly_run(fi, fixture_run, int(row["capacity_override"]), True)
                fixture_entries.extend(ent); fixture_summaries.append(summ)
                fixture_cash = summ["ending_cash"]
            st.session_state.fixture_ledger = fixture_entries
            st.session_state.fixture_summaries = fixture_summaries
            st.session_state.fixture_run_id = fixture_run
            st.success(f"Fixture replay completed in separate run {fixture_run}.")
        if st.session_state.get("fixture_summaries"):
            st.dataframe(pd.DataFrame([{k: str(v) for k,v in s.items() if k in ("month","requested","capacity","booked","attended","sold","unserved_demand","revenue","marketing","operating_profit","ending_cash")} for s in st.session_state.fixture_summaries]), use_container_width=True)

        st.divider()
        st.subheader("Dated Event Ledger")
        ledger = st.session_state.ledger
        freq = st.selectbox("Aggregate ledger view", ["Daily", "Monthly", "Yearly"], index=1)
        st.dataframe(aggregate_ledger(ledger, freq), use_container_width=True)
        if ledger:
            st.dataframe(pd.DataFrame(ledger), use_container_width=True)
            st.download_button("Download ledger CSV", pd.DataFrame(ledger).to_csv(index=False).encode(), "event_ledger.csv", "text/csv")
        else:
            st.info("No normal-mode transactions yet. Run a simulation to create dated events.")

    elif page == "Save / Load & Exports":
        st.header("Durable Save / Load JSON Snapshot")
        inputs = get_inputs_from_session()
        current_snapshot = snapshot_from_state(inputs, st.session_state.ledger)
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        st.download_button("Download current snapshot JSON", json.dumps(current_snapshot, indent=2).encode(),
                           "future_nurses_snapshot.json", "application/json")
        if st.button("Save snapshot to local snapshots/ folder"):
            path = SNAPSHOT_DIR / f"{inputs['snapshot_id']}.json"
            path.write_text(json.dumps(current_snapshot, indent=2), encoding="utf-8")
            st.success(f"Saved locally to {path}")
        uploaded = st.file_uploader("Load a JSON snapshot", type=["json"])
        if uploaded is not None and st.button("Validate and load snapshot"):
            try:
                candidate = json.loads(uploaded.getvalue().decode("utf-8"))
                validate_snapshot(candidate)
                # Idempotency: merge only transactions not already present by transaction_id.
                known = {e["transaction_id"] for e in st.session_state.ledger}
                st.session_state.ledger.extend([e for e in candidate["ledger"] if e["transaction_id"] not in known])
                st.session_state.snapshot_id = candidate["snapshot_id"]
                restored = candidate.get("assumptions", {})
                # Restore only known editable input keys; team/course/version remain locked constants.
                for key in ("business_name", "selected_level", "simulation_date", "period_unit",
                            "random_seed", "working_days", "base_price", "D0", "staff_hours",
                            "hours_per_unit", "venue_seats", "no_show_rate", "satisfaction_mean",
                            "owner_contribution", "equipment_assets", "preopening_expense",
                            "opening_inventory", "opening_cash", "opening_retained_earnings",
                            "refund_amount", "loan_principal", "interest", "additional_capex",
                            "new_owner_cash", "principal_repayment", "enable_depreciation",
                            "depreciation", "teacher_approval"):
                    if key in restored:
                        value = restored[key]
                        st.session_state[key] = date.fromisoformat(value) if key == "simulation_date" else value
                for key, value in restored.get("variable_costs", {}).items():
                    if key in DEFAULTS["variable_costs"]: st.session_state[f"var_{key}"] = value
                for key, value in restored.get("fixed_costs", {}).items():
                    if key in DEFAULTS["fixed_costs"]: st.session_state[f"fixed_{key}"] = value
                st.success("Snapshot validated and loaded. Existing transaction IDs were merged without duplication.")
            except Exception as exc:
                st.error(f"Snapshot rejected: {exc}")
        st.subheader("Run protection")
        st.write("Each explicit run receives a unique run_id. Browser refreshes do not execute a simulation. Replaying the same saved ledger merges by transaction_id and does not duplicate existing entries.")
        st.write(f"Normal ledger transactions: {len(st.session_state.ledger)}")
        st.write(f"Unique run IDs this session: {len(st.session_state.run_ids)}")
        st.download_button("Export normal ledger CSV",
            pd.DataFrame(st.session_state.ledger).to_csv(index=False).encode() if st.session_state.ledger else b"",
            "future_nurses_ledger.csv", "text/csv")

    else:
        st.header(page)
        st.info("Planned stage placeholder. This section is intentionally not simulated in Stage 1.")
        if "Stage 2" in page:
            st.write("Planned: Department Handoffs, Full Report Center, Costed Marketing Studio.")
        elif "Stage 3" in page:
            st.write("Planned: Assessments and Cross-Course Connections.")
        else:
            st.write("Planned: Optional Storefront. No checkout, public listing, or publishing is enabled.")

if __name__ == "__main__":
    main()
