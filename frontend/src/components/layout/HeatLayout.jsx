import { AlertTriangle, BarChart3, Brain, Database, LayoutDashboard, LineChart, Map, MapPin, Settings, SlidersHorizontal, TrendingUp, Thermometer, Workflow } from 'lucide-react'
import ModuleLayout from './ModuleLayout'
import { UrbanHeatProvider } from '../../features/urbanHeat/context/UrbanHeatContext'

const navItems = [
  { to: '/heat/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/heat/map', label: 'Historical Heat', icon: Map },
  { to: '/heat/trends', label: 'Trend Analysis', icon: LineChart },
  { to: '/heat/forecast', label: 'Future Forecast', icon: TrendingUp },
  { to: '/heat/scenario', label: 'Scenario Simulator', icon: SlidersHorizontal },
  { to: '/heat/xai', label: 'Explainable AI', icon: Brain },
  { to: '/heat/priority', label: 'Cooling Priority', icon: AlertTriangle },
  { to: '/heat/zone', label: 'Grid Details', icon: MapPin },
  { to: '/heat/areas', label: 'Area Comparison', icon: BarChart3 },
  { to: '/heat/dataset', label: 'Data Status', icon: Database },
  { to: '/heat/methodology', label: 'Methodology', icon: Workflow },
  { to: '/heat/settings', label: 'Settings', icon: Settings },
]

export default function HeatLayout() {
  return <UrbanHeatProvider>
    <ModuleLayout
      navItems={navItems}
      brand="KaduwelaHeat-XAI"
      subtitle="Urban Heat Intelligence"
      icon={Thermometer}
      headerTitle="Urban Heat Intelligence"
      headerSubtitle="Kaduwela · Backend-connected research dashboard"
      footer={<><b className="text-white/75">Urban Heat research component</b><br/>Real Step 34 dashboard outputs · forecasts include uncertainty.</>}
      accent="#2DD4BF"
      searchable={false}
    />
  </UrbanHeatProvider>
}
