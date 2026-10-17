import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import Toast from './components/shared/Toast'
import Login from './pages/common/Login'
import ProjectHome from './pages/common/ProjectHome'
import HeatLayout from './components/layout/HeatLayout'

import HeatDashboard from './pages/urbanHeat/Dashboard'
import HeatMap from './pages/urbanHeat/HeatMap'
import TrendAnalysis from './pages/urbanHeat/TrendAnalysis'
import Forecast from './pages/urbanHeat/Forecast'
import HeatScenario from './pages/urbanHeat/ScenarioSimulator'
import ExplainableAI from './pages/urbanHeat/ExplainableAI'
import CoolingPriority from './pages/urbanHeat/CoolingPriority'
import ZoneDetails from './pages/urbanHeat/ZoneDetails'
import AreaComparison from './pages/urbanHeat/AreaComparison'
import Dataset from './pages/urbanHeat/Dataset'
import Settings from './pages/urbanHeat/Settings'
import HeatMethodology from './pages/urbanHeat/Methodology'

function Guard({ children }) {
  const location = useLocation()
  const authed = localStorage.getItem('gbuei_auth') === '1'
  if (!authed) return <Navigate to="/" replace state={{ from: location.pathname }}/>
  return children
}

export default function App() {
  return <><Routes>
    <Route path="/" element={<Login/>}/>
    <Route path="/project-home" element={<Guard><ProjectHome/></Guard>}/>

    <Route path="/heat" element={<Guard><HeatLayout/></Guard>}>
      <Route index element={<Navigate to="dashboard" replace/>}/>
      <Route path="dashboard" element={<HeatDashboard/>}/>
      <Route path="map" element={<HeatMap/>}/>
      <Route path="trends" element={<TrendAnalysis/>}/>
      <Route path="forecast" element={<Forecast/>}/>
      <Route path="scenario" element={<HeatScenario/>}/>
      <Route path="xai" element={<ExplainableAI/>}/>
      <Route path="priority" element={<CoolingPriority/>}/>
      <Route path="zone" element={<ZoneDetails/>}/>
      <Route path="areas" element={<AreaComparison/>}/>
      <Route path="dataset" element={<Dataset/>}/>
      <Route path="methodology" element={<HeatMethodology/>}/>
      <Route path="settings" element={<Settings/>}/>
    </Route>

    <Route path="*" element={<Navigate to="/project-home" replace/>}/>
  </Routes><Toast/></>
}
