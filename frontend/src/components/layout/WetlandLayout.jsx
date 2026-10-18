
import {
  LayoutDashboard,
  History,
  Search,
  CalendarRange,
  GitCompareArrows,
  Waves,
  Layers,
  MapPin,
  BookOpen,
  Droplets,
} from 'lucide-react'

import ModuleLayout from './ModuleLayout'

const navItems = [
  {
    to: '/wetlands/dashboard',
    label: 'Overview Dashboard',
    icon: LayoutDashboard,
  },
  {
    to: '/wetlands/historical',
    label: 'Historical Wetland Change',
    icon: History,
  },
  {
    to: '/wetlands/individual',
    label: 'Individual Wetland Explorer',
    icon: History,
  },
  {
    to: '/wetlands/candidates',
    label: 'Candidate Wetlands',
    icon: Search,
  },
  {
    to: '/wetlands/future',
    label: 'Future Wetland Scenarios',
    icon: CalendarRange,
  },
  {
    to: '/wetlands/future-comparison',
    label: 'Local 2030 Scenario',
    icon: GitCompareArrows,
  },
  {
    to: '/wetlands/waterlogging',
    label: 'Waterlogging Susceptibility',
    icon: Waves,
  },
  {
    to: '/wetlands/scenarios',
    label: 'Scenario Comparison',
    icon: Layers,
  },
  {
    to: '/wetlands/gn-explorer',
    label: 'GN Division Explorer',
    icon: MapPin,
  },
  {
    to: '/wetlands/methodology',
    label: 'Methodology & Data Status',
    icon: BookOpen,
  },
]

export default function WetlandLayout() {
  return (
    <ModuleLayout
      navItems={navItems}
      brand="Kaduwela Wetland"
      subtitle="Wetland Intelligence"
      icon={Droplets}
      headerTitle="Wetland Intelligence"
      headerSubtitle="Kaduwela - Wetland Change and Waterlogging Research"
      footer={
        <>
          <b className="text-white/75">
            Wetland Research Component
          </b>
          <p className="mt-1">
            Historical change, candidate screening,
            exploratory scenarios and relative
            waterlogging susceptibility.
          </p>
        </>
      }
      accent="#2DD4BF"
      searchable={false}
    />
  )
}
