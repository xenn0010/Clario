# Clario Frontend Experience

A bespoke frontend for Clario's autonomous meeting platform. Built with Vite + React + TypeScript, animated with Framer Motion, and styled with Tailwind for a fluid, curved aesthetic that pairs with the FastAPI backend.

## Getting Started

1. Install dependencies:

   ```bash
   cd frontend
   npm install
   ```

2. Run the development server (proxies to the FastAPI backend on port 8000 by default):

   ```bash
   npm run dev
   ```

   Set `VITE_API_BASE_URL` in a `.env.local` file if your backend runs elsewhere.

3. Build for production:

   ```bash
   npm run build
   npm run preview
   ```

## Experience Highlights

- **Hero Arc:** Curved hero section with animated orbit of core agents and gradient glows.
- **Agent Operations Cockpit:** Live agent roster, discourse feed, and meeting cadence panels using backend data (with graceful fallbacks).
- **Graph Studio:** d3-force driven Neo4j simulation with scenario toggles and focused highlights.
- **Design Language:** Deep midnight palette with lagoon, seafoam, and sunburst accents—no purple in sight.

## Backend Integration

- Agents: `/api/v1/agents/organizations/{org_id}/agents`
- Active Meetings: `/api/v1/agents/meetings/active`
- Graph Insights: `/api/v1/graph/organizations/{org_id}/insights`
- Graph Patterns: `/api/v1/graph/organizations/{org_id}/patterns`

All requests default to `/api/v1` when `VITE_API_BASE_URL` is not set so the Vite dev proxy can forward to `localhost:8000`.
