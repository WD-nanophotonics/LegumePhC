# LegumePhC Studio — web frontend

React + TypeScript + Plotly UI for the localhost backend in
`src/legumephc/server/`. `npm run build` type-checks and writes the bundle to
`src/legumephc/server/static/`, which the backend serves at `/`.

```powershell
npm ci
npm run dev     # :5173, proxies /api to 127.0.0.1:8765 (start `python -m legumephc.server --no-browser`)
npm run build
```

## Layout

| Path | Role |
|---|---|
| `src/api.ts`, `src/types.ts` | Typed client for `/api/*` and mirrors of the project schema |
| `src/operations/types.ts` | `OperationDef`, `FieldSpec`, `StyleSpec` contracts |
| `src/operations/definitions.ts` | **The operation registry** — one entry per calculation |
| `src/operations/renderers.ts` | Plot families (`renderBandLines`, `renderMap`) |
| `src/components/SpecForm.tsx` | Generic form renderers for field/style specs |
| `src/components/Dialog.tsx` | In-app prompt/confirm dialogs |
| `src/App.tsx` | Project state, run lifecycle (SSE), layout |

## Adding a calculation

1. Backend: implement it in `legumephc.studio.worker.execute_request`, add it
   to `legumephc.studio.forms.OPERATIONS` and to `project._validate_calculation`,
   and give `legumephc.studio.inspection.record_view` its raw rows.
2. Frontend: add one `OperationDef` to `src/operations/definitions.ts`
   (form fields, plot-style controls, a renderer, and a tree label).
   Forms, the plot-style panel and the result plot are generated from it.

Scientific rules still apply in the UI: plot only raw samples, never raw
complex eigenvectors, and show qualification labels as recorded.
