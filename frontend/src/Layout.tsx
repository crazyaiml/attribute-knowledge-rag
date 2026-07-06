import { NavLink, Outlet } from "react-router-dom";
import { ProviderBar } from "./components/ProviderBar";
import type { ProvidersHealth } from "./types";

const NAV_ITEMS = [
  { to: "/", label: "Demo", end: true },
  { to: "/catalog", label: "Catalog" },
  { to: "/settings", label: "Settings" },
];

export function Layout({ providers }: { providers: ProvidersHealth | null }) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>AK-RAG</h1>
          <p className="tagline">
            Attribute Knowledge RAG — a deterministic agentic workflow for translating natural
            language into governed enterprise attributes.
          </p>
        </div>
        <ProviderBar providers={providers} />
      </header>

      <nav className="app-nav">
        {NAV_ITEMS.map(({ to, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `nav-link${isActive ? " nav-link--active" : ""}`}
          >
            {label}
          </NavLink>
        ))}
      </nav>

      <Outlet />
    </div>
  );
}
