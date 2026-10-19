export const greenPages = {
  "dashboard": {
    "slug": "dashboard",
    "label": "Dashboard",
    "title": "Urban Green Intelligence Dashboard",
    "description": "An integrated overview of historical green-cover change, urbanization pressure, development suitability and conservation intelligence.",
    "planned": [
      "Historical green-cover and built-up indicators",
      "Research summary charts and study-area map",
      "Suitability, conservation and planning summaries"
    ],
    "endpoints": [
      "/historical/summary",
      "/pressure/overview",
      "/suitability/overview",
      "/conservation/overview"
    ]
  },
  "greenCover": {
    "slug": "green-cover",
    "label": "Green-Cover Analysis",
    "title": "Historical Green-Cover Analysis",
    "description": "Observed historical land-cover patterns and annual green-cover evidence for the Kaduwela study area.",
    "planned": [
      "Historical land-cover map selection",
      "Annual class-area trends",
      "Core analysis period and data-quality indicators"
    ],
    "endpoints": [
      "/historical/class-areas",
      "/historical/maps",
      "/historical/annual-qa"
    ]
  },
  "transitions": {
    "slug": "transitions",
    "label": "Green-Built Transitions",
    "title": "Green-Built Transition Analysis",
    "description": "Historical green-to-built conversion evidence, transition comparisons and uncertainty-aware interpretation.",
    "planned": [
      "Historical transition maps",
      "Robust and strict transition evidence",
      "Transition comparisons and confidence information"
    ],
    "endpoints": [
      "/historical/transitions/analysis",
      "/historical/transitions/comparison",
      "/historical/transitions/confidence"
    ]
  },
  "pressure": {
    "slug": "pressure",
    "label": "Urbanization Pressure",
    "title": "Urbanization Pressure Intelligence",
    "description": "Relative model-derived urbanization pressure for the combined 2026?2028 future assessment period.",
    "planned": [
      "Interactive grid-level pressure map",
      "Pressure scores, flags and model coverage",
      "Grid selection and out-of-domain information"
    ],
    "endpoints": [
      "/pressure/overview",
      "/pressure/map",
      "/pressure/grids"
    ]
  },
  "xai": {
    "slug": "xai",
    "label": "Explainable AI",
    "title": "Urbanization Pressure Explainable AI",
    "description": "SHAP-based model explanation, feature importance and domain-level interpretation.",
    "planned": [
      "Feature importance charts",
      "Explanatory domain summaries",
      "Research interpretation and limitations"
    ],
    "endpoints": [
      "/pressure/xai/overview",
      "/pressure/xai/features",
      "/pressure/xai/domains"
    ]
  },
  "suitability": {
    "slug": "suitability",
    "label": "Development Suitability",
    "title": "Expert-AHP Development Suitability",
    "description": "Continuous GIS-MCDA suitability scores derived from the approved expert-AHP methodology.",
    "planned": [
      "Continuous suitability GIS map",
      "AHP weights and consistency details",
      "Grid-level criterion and suitability evidence"
    ],
    "endpoints": [
      "/suitability/overview",
      "/suitability/map",
      "/suitability/grids"
    ]
  },
  "conservation": {
    "slug": "conservation",
    "label": "Conservation Intelligence",
    "title": "Green Conservation Intelligence",
    "description": "Green Importance and Conservation Priority research outputs, including explicit missing-data coverage.",
    "planned": [
      "Conservation Priority map",
      "Green Importance summaries",
      "Scored and unscored domain explanations"
    ],
    "endpoints": [
      "/conservation/overview",
      "/conservation/map",
      "/conservation/grids"
    ]
  },
  "planning": {
    "slug": "planning-conflict",
    "label": "Planning Conflict",
    "title": "Planning Conflict and Interventions",
    "description": "Development conflict evidence, separate conservation concerns and explainable planning-review recommendations.",
    "planned": [
      "Planning conflict grid map",
      "Review rules and intervention explanations",
      "Separate Development Conflict and Conservation Priority evidence"
    ],
    "endpoints": [
      "/planning/grids",
      "/planning/map"
    ]
  },
  "gnHotspots": {
    "slug": "gn-hotspots",
    "label": "GN & Hotspot Explorer",
    "title": "GN Division and Hotspot Explorer",
    "description": "GN-level area-weighted summaries and exploratory spatial screening groups.",
    "planned": [
      "GN division selection and comparison",
      "GN map and planning summaries",
      "Candidate hotspot maps and review tables"
    ],
    "endpoints": [
      "/planning/gn-divisions",
      "/planning/gn-map",
      "/planning/hotspots",
      "/planning/hotspot-map"
    ]
  },
  "methodology": {
    "slug": "methodology",
    "label": "Methodology & Data QA",
    "title": "Research Methodology and Data Quality",
    "description": "Research methods, data coverage, model scope, quality assurance and documented limitations.",
    "planned": [
      "Research pipeline and data sources",
      "Model and AHP methodology summaries",
      "Backend data-health and research limitations"
    ],
    "endpoints": [
      "/metadata",
      "/manifest",
      "/data-health",
      "/historical/annual-qa"
    ]
  }
}
