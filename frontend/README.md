# Green-Blue UEI Frontend — Urban Heat + Shared Shell

This cleaned package intentionally contains the complete Urban Heat frontend plus common/shared project UI only. Prototype implementations for Urbanization, Wetland, and River Corridor were removed at the team's request so those members can integrate their own work independently.

- `backend/` and `Urban_Heat_Research/` remain external and unchanged.
- Urban Heat remains under `src/pages/urbanHeat/` and `src/features/urbanHeat/`.
- Shared/common UI remains under `src/components/`, `src/pages/common/`, and `src/contexts/`.
- Other member cards on Project Home are intentionally disabled until their real frontend routes are added.

# Green-Blue UEI Frontend

React + Vite + Tailwind frontend for the Green-Blue Urban Environmental Intelligence group research project.

## Structure

- `src/pages/common/` — shared login/project-home pages.
- `src/pages/urbanization/` — Component 1 pages.
- `src/pages/urbanHeat/` — Component 2 Urban Heat pages.
- `src/pages/wetland/` — Component 3 pages.
- `src/pages/riverCorridor/` — Component 4 pages.
- `src/features/urbanHeat/` — Urban Heat API, context, map and reusable feature logic.
- `src/components/` — shared layout/UI components.

The Urban Heat module is connected to the supplied FastAPI backend. The other three modules retain the supplied prototype/reference implementation until their respective backend APIs are integrated.

## Expected sibling project layout

```text
J26-DS-338/
├── backend/
├── Urban_Heat_Research/
└── frontend/
```

The frontend never writes to `backend/` or `Urban_Heat_Research/`.

## Install

```powershell
cd frontend
npm install
```

Create `.env` only if the backend is not using the default local address:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Run backend (from project root)

```powershell
python -m uvicorn backend.app:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Run frontend

```powershell
cd frontend
npm run dev
```

Vite normally opens at `http://localhost:5173`.

## Kaduwela boundary

Before `dev` and `build`, `scripts/prepare-boundary.mjs` looks for the existing read-only official KMC boundary at:

```text
../Urban_Heat_Research/data/raw/boundary/kaduwela_kmc_boundary.kml
```

When it exists, the script converts that KML into a frontend-only GeoJSON file under `public/data/`. The research KML is not edited. If it is unavailable, the frontend falls back to a study-area outline derived from the supplied real WGS84 grid geometry.

## Build

```powershell
npm run build
```

The production build is written to `dist/` and should not be committed unless your team explicitly wants build artifacts in Git.
