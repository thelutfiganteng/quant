"""
Financial Astrology & Astronomical Pivot Engine.

This module implements quantitative astro-finance based on Gann's methods,
without requiring external C-libraries. It uses established astronomical
approximations to calculate phases and retrograde periods.

Algorithms included:
1. Lunar Cycle Analysis: New Moon and Full Moon align with short-term market reversals.
2. Mercury Retrograde Approximation: Erratic market behavior and false breakouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AstroPivotResult:
    date: datetime
    pivot_probability: float
    primary_factors: list[str]
    moon_phase_pct: float
    is_mercury_retrograde: bool
    summary: str


class AstroPivotEngine:
    """
    Engine to compute astronomical events and generate market pivot scores
    using pure mathematical approximations for speed and compatibility.
    """

    def __init__(self):
        # Known reference dates
        self.known_new_moon = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
        self.lunar_month = 29.53058770576
        
        # Mercury synodic period is ~115.88 days. Retrograde lasts ~21 days.
        # Known retrograde start: 2023-12-13
        self.known_mercury_retrograde = datetime(2023, 12, 13, tzinfo=timezone.utc)
        self.mercury_synodic = 115.88
        self.mercury_retro_duration = 21.0

    def get_moon_phase(self, target_date: datetime) -> float:
        """Calculate moon phase as a percentage (0.0 to 1.0)."""
        diff = (target_date - self.known_new_moon).total_seconds() / 86400.0
        cycles = diff / self.lunar_month
        phase = cycles % 1.0
        return phase

    def is_mercury_retrograde(self, target_date: datetime) -> bool:
        """Approximate if Mercury is retrograde."""
        diff = (target_date - self.known_mercury_retrograde).total_seconds() / 86400.0
        cycles = diff / self.mercury_synodic
        cycle_position = cycles % 1.0
        # Retrograde happens in the first 21 days of the 115.88 day cycle
        days_into_cycle = cycle_position * self.mercury_synodic
        return days_into_cycle <= self.mercury_retro_duration

    def evaluate_date(self, target_date: datetime) -> AstroPivotResult:
        """
        Evaluate a date for market pivot probability based on astro algorithms.
        """
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=timezone.utc)
            
        factors = []
        score = 0.0
        
        # Helper to detect local phase peaks to avoid multi-day duplicate triggers
        def get_phase_dist(date_val: datetime, target: float) -> float:
            p = self.get_moon_phase(date_val)
            if target == 0.0:
                return min(p, 1.0 - p)
            return abs(p - target)

        def is_phase_peak(target: float) -> bool:
            dist_today = get_phase_dist(target_date, target)
            dist_yesterday = get_phase_dist(target_date - timedelta(days=1), target)
            dist_tomorrow = get_phase_dist(target_date + timedelta(days=1), target)
            # Must be the absolute closest day to the exact phase within a +/- 1 day window
            return dist_today <= dist_yesterday and dist_today <= dist_tomorrow and dist_today < 0.035

        # 1. Lunar Phase Analysis
        moon_phase = self.get_moon_phase(target_date)
        
        if is_phase_peak(0.0):
            score += 0.40
            factors.append("New Moon (High Pivot Probability)")
        elif is_phase_peak(0.5):
            score += 0.40
            factors.append("Full Moon (High Pivot Probability)")
        elif is_phase_peak(0.25):
            score += 0.15
            factors.append("First Quarter Moon (Trend Acceleration)")
        elif is_phase_peak(0.75):
            score += 0.15
            factors.append("Last Quarter Moon (Trend Acceleration)")
            
        # 2. Mercury Retrograde Analysis
        mercury_retro = self.is_mercury_retrograde(target_date)
        if mercury_retro:
            score += 0.20
            # Retrograde itself is a continuous state, we just add it as a supporting factor
            # It provides high volatility background context.
            factors.append("Mercury Retrograde (Volatility Context)")
            
            # Add extra points if Mercury Retrograde starts or ends exactly today
            was_retro = self.is_mercury_retrograde(target_date - timedelta(days=1))
            will_be_retro = self.is_mercury_retrograde(target_date + timedelta(days=1))
            if mercury_retro and not was_retro:
                score += 0.25
                factors.append("Mercury Retrograde START (Major Reversal)")
            elif not mercury_retro and was_retro: # Wait, this won't hit because if mercury_retro is false, it skips this block.
                pass

        # Check for Retrograde END
        if not mercury_retro:
            was_retro = self.is_mercury_retrograde(target_date - timedelta(days=1))
            if was_retro:
                score += 0.25
                factors.append("Mercury Retrograde END (Major Reversal)")
            
        # 3. Time-based harmonic convergence (Gann 90-day cycles)
        day_of_year = target_date.timetuple().tm_yday
        # Approximate: Mar 20, Jun 21, Sep 23, Dec 21
        gann_days = [79, 172, 266, 355]
        for gd in gann_days:
            if day_of_year == gd:
                score += 0.35
                factors.append("Gann Seasonal Pivot (Equinox/Solstice Exact)")
                break

        # Normalize score
        pivot_prob = min(round(score, 4), 0.95)
        
        if not factors or pivot_prob < 0.1:
            factors = ["No major astro alignments"]
            pivot_prob = 0.05
            
        summary = f"Probability: {pivot_prob*100:.0f}%. Events: {', '.join(factors) if factors[0] != 'No major astro alignments' else 'None'}"
        
        return AstroPivotResult(
            date=target_date,
            pivot_probability=pivot_prob,
            primary_factors=factors,
            moon_phase_pct=moon_phase,
            is_mercury_retrograde=mercury_retro,
            summary=summary
        )

    def generate_calendar(self, start_date: datetime, days: int = 30) -> pd.DataFrame:
        """Scan a forward period to identify the highest probability pivot dates."""
        results = []
        for i in range(days):
            current = start_date + timedelta(days=i)
            # Evaluate at market close (e.g. 16:00 UTC)
            eval_time = current.replace(hour=16, minute=0, second=0, microsecond=0)
            res = self.evaluate_date(eval_time)
            
            # Only include dates that have actual pivot factors (score >= 0.15)
            # This suppresses normal non-event days and prevents clutter
            if res.pivot_probability >= 0.15:
                results.append({
                    "date": current.strftime("%Y-%m-%d"),
                    "pivot_score": res.pivot_probability,
                    "is_retrograde": res.is_mercury_retrograde,
                    "factors": res.primary_factors,
                    "summary": res.summary
                })
            
        df = pd.DataFrame(results)
        if not df.empty:
            return df.sort_values(by="date", ascending=True)
        return df
