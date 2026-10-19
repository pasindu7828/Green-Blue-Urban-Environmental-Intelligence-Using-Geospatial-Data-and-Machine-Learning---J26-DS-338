import {
  AlertTriangle,
  BarChart3,
  Brain,
  Database,
  LayoutDashboard,
  Leaf,
  Map,
  MapPin,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react'

import ModuleLayout from '../../../components/layout/ModuleLayout'
import { UrbanGreenProvider } from '../context/UrbanGreenContext'
import { greenPages } from '../config/pages'

const pageIcons = {
  dashboard: LayoutDashboard,
  greenCover: Leaf,
  transitions: BarChart3,
  pressure: TrendingUp,
  xai: Brain,
  suitability: Map,
  conservation: ShieldCheck,
  planning: AlertTriangle,
  gnHotspots: MapPin,
  methodology: Database,
}

const navItems = Object.entries(greenPages).map(([key, page]) => ({
  to: `/green/${page.slug}`,
  label: page.label,
  icon: pageIcons[key],
}))

export default function UrbanGreenLayout() {
  return (
    <UrbanGreenProvider>
      <ModuleLayout
        navItems={navItems}
        brand="Kaduwela Green-XAI"
        subtitle="Urban Green Intelligence"
        icon={Leaf}
        headerTitle="Urban Green Intelligence"
        headerSubtitle="Kaduwela | Research-backed planning intelligence"        footer={
          <>
            <b className="text-white/75">Urban Green research component</b>
            <br />
            Historical change ? Urban pressure ? GIS decision support
          </>
        }
        accent="#2DD4BF"
        searchable={false}
      />
    </UrbanGreenProvider>
  )
}
