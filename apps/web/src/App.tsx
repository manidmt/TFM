import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import AdminRoute from './components/AdminRoute';

import Landing from './pages/Landing';
import Predictions from './pages/Predictions';
import AssetHistory from './pages/AssetHistory';
import Login from './pages/Login';
import PortfolioList from './pages/app/PortfolioList';
import PortfolioDetail from './pages/app/PortfolioDetail';
import OpsOverview from './pages/ops/OpsOverview';
import OpsAssets from './pages/ops/OpsAssets';
import OpsPromotions from './pages/ops/OpsPromotions';

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Layout>
          <Routes>
            {/* Public */}
            <Route path="/" element={<Landing />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/predictions/:assetId" element={<AssetHistory />} />
            <Route path="/login" element={<Login />} />

            {/* Private — authenticated users */}
            <Route
              path="/app"
              element={<ProtectedRoute><PortfolioList /></ProtectedRoute>}
            />
            <Route
              path="/app/portfolios/:portfolioId"
              element={<ProtectedRoute><PortfolioDetail /></ProtectedRoute>}
            />

            {/* Admin only */}
            <Route
              path="/ops"
              element={<AdminRoute><OpsOverview /></AdminRoute>}
            />
            <Route
              path="/ops/assets"
              element={<AdminRoute><OpsAssets /></AdminRoute>}
            />
            <Route
              path="/ops/promotions"
              element={<AdminRoute><OpsPromotions /></AdminRoute>}
            />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
      </AuthProvider>
    </BrowserRouter>
  );
}
