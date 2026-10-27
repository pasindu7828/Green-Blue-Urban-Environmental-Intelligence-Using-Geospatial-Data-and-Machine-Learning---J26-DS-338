
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

// Urban Green Intelligence — preserve all 10 research pages.
import UrbanGreenLayout from './features/urbanGreen/components/UrbanGreenLayout'
import GreenDashboard from './pages/urbanGreen/Dashboard'
import GreenGreenCover from './pages/urbanGreen/GreenCover'
import GreenTransitions from './pages/urbanGreen/Transitions'
import GreenUrbanizationPressure from './pages/urbanGreen/UrbanizationPressure'
import GreenExplainableAI from './pages/urbanGreen/ExplainableAI'
import GreenDevelopmentSuitability from './pages/urbanGreen/DevelopmentSuitability'
import GreenConservationIntelligence from './pages/urbanGreen/ConservationIntelligence'
import GreenPlanningConflict from './pages/urbanGreen/PlanningConflict'
import GreenGnHotspotExplorer from './pages/urbanGreen/GnHotspotExplorer'
import GreenMethodology from './pages/urbanGreen/Methodology'

// Wetland Intelligence — preserve teammates' 7 research pages.
import WetlandLayout from './components/layout/WetlandLayout'
import WetlandDashboard from './pages/wetland/Dashboard'
import HistoricalChange from './pages/wetland/HistoricalChange'
import IndividualWetlandExplorer from './pages/wetland/IndividualWetlandExplorer'
import CandidateWetlands from './pages/wetland/CandidateWetlands'
import FutureWetlandPrediction from './pages/wetland/FutureWetlandPrediction'
import IndividualWetlandFutureComparison from './pages/wetland/IndividualWetlandFutureComparison'
import SeasonalWaterlogging from './pages/wetland/SeasonalWaterlogging'

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

    {/* Existing Urban Heat routes */}
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

    {/* Urban Green Intelligence routes */}
    <Route path="/green" element={<Guard><UrbanGreenLayout/></Guard>}>
      <Route index element={<Navigate to="dashboard" replace/>}/>
      <Route path="dashboard" element={<GreenDashboard/>}/>
      <Route path="green-cover" element={<GreenGreenCover/>}/>
      <Route path="transitions" element={<GreenTransitions/>}/>
      <Route path="pressure" element={<GreenUrbanizationPressure/>}/>
      <Route path="xai" element={<GreenExplainableAI/>}/>
      <Route path="suitability" element={<GreenDevelopmentSuitability/>}/>
      <Route path="conservation" element={<GreenConservationIntelligence/>}/>
      <Route path="planning-conflict" element={<GreenPlanningConflict/>}/>
      <Route path="gn-hotspots" element={<GreenGnHotspotExplorer/>}/>
      <Route path="methodology" element={<GreenMethodology/>}/>
    </Route>

    {/* Wetland Intelligence routes */}
    <Route path="/wetlands" element={<Guard><WetlandLayout/></Guard>}>
      <Route index element={<Navigate to="dashboard" replace/>}/>
      <Route path="dashboard" element={<WetlandDashboard/>}/>
      <Route path="historical" element={<HistoricalChange/>}/>
      <Route path="individual" element={<IndividualWetlandExplorer/>}/>
      <Route path="candidates" element={<CandidateWetlands/>}/>
      <Route path="future" element={<FutureWetlandPrediction/>}/>
      <Route path="future-comparison" element={<IndividualWetlandFutureComparison/>}/>
      <Route path="waterlogging" element={<SeasonalWaterlogging/>}/>
    </Route>

    <Route path="*" element={<Navigate to="/project-home" replace/>}/>
  </Routes><Toast/></>
}
