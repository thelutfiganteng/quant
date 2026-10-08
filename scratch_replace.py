import re

with open('d:/QuantDashboard/src/dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace DEMO_RELEASES
content = re.sub(
    r'DEMO_RELEASES = \[.*?\]\n\nDEMO_CALENDAR',
    '''DEMO_RELEASES = [
    {
        "indicator": "NFP",
        "release_date": "2026-10-02T12:30:00Z",
        "actual": 185,
        "consensus": 170,
        "previous": 142,
        "surprise": 15,
        "surprise_zscore": 0.85,
        "source": "BLS",
    },
    {
        "indicator": "UNEMPLOYMENT_RATE",
        "release_date": "2026-10-02T12:30:00Z",
        "actual": 4.1,
        "consensus": 4.2,
        "previous": 4.2,
        "surprise": -0.1,
        "surprise_zscore": -1.2,
        "source": "BLS",
    },
    {
        "indicator": "ISM_MANUFACTURING",
        "release_date": "2026-10-01T14:00:00Z",
        "actual": 47.2,
        "consensus": 47.5,
        "previous": 47.2,
        "surprise": -0.3,
        "surprise_zscore": -0.4,
        "source": "ISM",
    },
    {
        "indicator": "PCE",
        "release_date": "2026-09-25T12:30:00Z",
        "actual": 2.2,
        "consensus": 2.3,
        "previous": 2.5,
        "surprise": -0.1,
        "surprise_zscore": -0.9,
        "source": "BEA",
    },
    {
        "indicator": "FOMC Rate Decision",
        "release_date": "2026-09-16T18:00:00Z",
        "actual": 5.00,
        "consensus": 5.00,
        "previous": 5.50,
        "surprise": 0.0,
        "surprise_zscore": 0.0,
        "source": "FED",
    },
    {
        "indicator": "RETAIL_SALES",
        "release_date": "2026-09-15T12:30:00Z",
        "actual": 0.1,
        "consensus": 0.2,
        "previous": 1.1,
        "surprise": -0.1,
        "surprise_zscore": -0.5,
        "source": "Census",
    },
    {
        "indicator": "CPI",
        "release_date": "2026-09-11T12:30:00Z",
        "actual": 2.5,
        "consensus": 2.5,
        "previous": 2.9,
        "surprise": 0.0,
        "surprise_zscore": 0.0,
        "source": "BLS",
    },
]

DEMO_CALENDAR''',
    content,
    flags=re.DOTALL
)

# Replace DEMO_CALENDAR
content = re.sub(
    r'DEMO_CALENDAR = \[.*?\]\n\nDEMO_NOWCASTS',
    '''DEMO_CALENDAR = [
    {
        "indicator": "CPI",
        "country": "US",
        "scheduled_date": "2026-10-14T12:30:00Z",
        "importance": "HIGH",
        "forecast": 2.3,
        "previous": 2.5,
    },
    {
        "indicator": "CORE_CPI",
        "country": "US",
        "scheduled_date": "2026-10-14T12:30:00Z",
        "importance": "HIGH",
        "forecast": 3.2,
        "previous": 3.2,
    },
    {
        "indicator": "RETAIL_SALES",
        "country": "US",
        "scheduled_date": "2026-10-15T12:30:00Z",
        "importance": "HIGH",
        "forecast": 0.3,
        "previous": 0.1,
    },
    {
        "indicator": "INITIAL_CLAIMS",
        "country": "US",
        "scheduled_date": "2026-10-15T12:30:00Z",
        "importance": "MEDIUM",
        "forecast": 220,
        "previous": 225,
    },
    {
        "indicator": "FOMC Rate Decision",
        "country": "US",
        "scheduled_date": "2026-11-04T18:00:00Z",
        "importance": "HIGH",
        "forecast": 4.75,
        "previous": 5.00,
    },
    {
        "indicator": "NFP",
        "country": "US",
        "scheduled_date": "2026-11-06T13:30:00Z",
        "importance": "HIGH",
        "forecast": 150,
        "previous": 185,
    },
]

DEMO_NOWCASTS''',
    content,
    flags=re.DOTALL
)

# Replace DEMO_NOWCASTS
content = re.sub(
    r'DEMO_NOWCASTS = \[.*?\]\n\nDEMO_SURPRISE_HISTORY',
    '''DEMO_NOWCASTS = [
    {
        "indicator": "CPI",
        "target_release_date": "2026-10-14T12:30:00Z",
        "estimated_at": "2026-10-08T06:00:00Z",
        "point_estimate": 2.4,
        "confidence_lower": 2.2,
        "confidence_upper": 2.6,
        "model_name": "cpi_bridge_v1",
        "consensus": 2.3,
    },
    {
        "indicator": "NFP",
        "target_release_date": "2026-11-06T13:30:00Z",
        "estimated_at": "2026-10-08T06:00:00Z",
        "point_estimate": 160,
        "confidence_lower": 130,
        "confidence_upper": 190,
        "model_name": "nfp_kalman_v1",
        "consensus": 150,
    },
    {
        "indicator": "GDP",
        "target_release_date": "2026-10-29T12:30:00Z",
        "estimated_at": "2026-10-08T06:00:00Z",
        "point_estimate": 2.7,
        "confidence_lower": 2.4,
        "confidence_upper": 3.0,
        "model_name": "gdp_bridge_v1",
        "consensus": 2.5,
    },
]

DEMO_SURPRISE_HISTORY''',
    content,
    flags=re.DOTALL
)

# Replace signals demo_data
content = re.sub(
    r'# Add historical signals \(past 30 days\).*?demo_data\.extend\(upcoming_events\)',
    '''from datetime import datetime, timezone
        
        upcoming_events = [
            ("NFP", 160, 150, "nfp_kalman_v1", datetime(2026, 11, 6, 13, 30, tzinfo=timezone.utc)),
            ("GDP", 2.7, 2.5, "gdp_bridge_v1", datetime(2026, 10, 29, 12, 30, tzinfo=timezone.utc)),
            ("CPI", 2.4, 2.3, "cpi_bridge_v1", datetime(2026, 10, 14, 12, 30, tzinfo=timezone.utc)),
            ("PCE", 2.2, 2.2, "pce_bridge_v1", datetime(2026, 10, 31, 12, 30, tzinfo=timezone.utc)),
            ("UNEMPLOYMENT_RATE", 4.1, 4.2, "unemp_kalman_v1", datetime(2026, 11, 6, 13, 30, tzinfo=timezone.utc)),
            ("PPI", 0.1, 0.2, "ppi_bridge_v1", datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)),
            ("RETAIL_SALES", 0.5, 0.3, "retail_kalman_v1", datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)),
            ("ISM_MANUFACTURING", 48.5, 47.8, "ism_mfg_bridge_v1", datetime(2026, 11, 2, 14, 0, tzinfo=timezone.utc)),
            ("INITIAL_CLAIMS", 215, 220, "claims_kalman_v1", datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)),
        ]
        
        # Historical events explicitly matched to DEMO_RELEASES
        historical_events = [
            ("NFP", 185, 170, "nfp_kalman_v1", datetime(2026, 10, 2, 12, 30, tzinfo=timezone.utc)),
            ("UNEMPLOYMENT_RATE", 4.1, 4.2, "unemp_kalman_v1", datetime(2026, 10, 2, 12, 30, tzinfo=timezone.utc)),
            ("ISM_MANUFACTURING", 47.2, 47.5, "ism_mfg_bridge_v1", datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc)),
            ("PCE", 2.2, 2.3, "pce_bridge_v1", datetime(2026, 9, 25, 12, 30, tzinfo=timezone.utc)),
            ("RETAIL_SALES", 0.1, 0.2, "retail_kalman_v1", datetime(2026, 9, 15, 12, 30, tzinfo=timezone.utc)),
            ("CPI", 2.5, 2.5, "cpi_bridge_v1", datetime(2026, 9, 11, 12, 30, tzinfo=timezone.utc)),
        ]

        demo_data = upcoming_events + historical_events''',
    content,
    flags=re.DOTALL
)

# Replace FOMC manual logic
content = re.sub(
    r'# Add explicit FOMC Signal aligned with the Calendar.*?if fomc_signal:',
    '''# Add explicit FOMC Signal aligned with the Calendar
        from datetime import timezone
        fomc_event_time = datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc)
        
        fomc_signal = gen.generate_fomc_signal(
            actual_rate=4.75,          
            consensus_rate=4.75,       
            recent_cpi_surprise=1.2,   
            event_time=fomc_event_time,
            asset="SPY"
        )
        if fomc_signal:''',
    content,
    flags=re.DOTALL
)

with open('d:/QuantDashboard/src/dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
