import { useCallback, useEffect, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./App.css";
import { getProviders } from "./api";
import { Layout } from "./Layout";
import { CatalogPage } from "./pages/CatalogPage";
import { DemoPage } from "./pages/DemoPage";
import { SettingsPage } from "./pages/SettingsPage";
import type { ProvidersHealth } from "./types";

export default function App() {
  const [providers, setProviders] = useState<ProvidersHealth | null>(null);

  const refreshProviders = useCallback(() => {
    getProviders()
      .then(setProviders)
      .catch(() => setProviders(null));
  }, []);

  useEffect(() => {
    refreshProviders();
  }, [refreshProviders]);

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout providers={providers} />}>
          <Route index element={<DemoPage providers={providers} />} />
          <Route path="catalog" element={<CatalogPage onCatalogChanged={refreshProviders} />} />
          <Route path="settings" element={<SettingsPage onProvidersChanged={refreshProviders} />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
