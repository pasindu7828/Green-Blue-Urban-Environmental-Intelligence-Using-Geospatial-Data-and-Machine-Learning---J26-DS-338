
import {
  Droplets,
  History,
  Search,
  CalendarRange,
  Waves,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { wetlandApi } from '../../features/wetland/services/wetlandApi'


const researchParts = [
  {
    title: 'Historical Wetland Change',
    description:
      'Explore monitored wetlands, historical observations and wetland area changes.',
    icon: History,
    part: 'Part 1',
    image: '/images/wetland/wetland1.jpg',
  },
  {
    title: 'Candidate Wetland Detection',
    description:
      'Explore screened candidate wetland areas that require further verification.',
    icon: Search,
    part: 'Part 2',
    image: '/images/wetland/wetland3.jpg',
  },
  {
    title: 'Future Wetland Scenarios',
    description:
      'Compare 2025 wetland distribution with exploratory 2030 gain, loss and persistence scenarios.',
    icon: CalendarRange,
    part: 'Part 3',
    image: '/images/wetland/wetland2.jpg',
  },
  {
    title: 'Waterlogging Susceptibility',
    description:
      'Explore relative waterlogging susceptibility maps, GN-level patterns and scenario uncertainty.',
    icon: Waves,
    part: 'Part 4',
    image: '/images/wetland/waterlogging1.jpg',
  },
]


export default function WetlandDashboard() {
  const [wetlandCount, setWetlandCount] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [candidateCount, setCandidateCount] = useState(null)
  const [candidateLoading, setCandidateLoading] = useState(true)
  const [candidateError, setCandidateError] = useState(null)
  const [futureGnCount, setFutureGnCount] = useState(null)
  const [futureLoading, setFutureLoading] = useState(true)
  const [futureError, setFutureError] = useState(null)
  const [highSusceptibilityArea, setHighSusceptibilityArea] = useState(null)
  const [highSusceptibilityShare, setHighSusceptibilityShare] = useState(null)
  const [waterloggingLoading, setWaterloggingLoading] = useState(true)
  const [waterloggingError, setWaterloggingError] = useState(null)

  useEffect(() => {
    const controller = new AbortController()

    async function loadData() {
      try {
        const result = await wetlandApi.historicalSummary(
          controller.signal
        )

        setWetlandCount(result.total_wetlands)
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      }
    }

    loadData()

    return () => controller.abort()
  }, [])

  // Part 2 - Primary Wetland Candidates
  useEffect(() => {
    const controller = new AbortController()

    async function loadCandidates() {
      try {
        const result = await wetlandApi.primaryCandidates(
          controller.signal
        )

        if (!Array.isArray(result.features)) {
          throw new Error('Unexpected candidate data format')
        }

        setCandidateCount(result.features.length)
      } catch (err) {
        if (err.name !== 'AbortError') {
          setCandidateError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setCandidateLoading(false)
        }
      }
    }

    loadCandidates()

    return () => controller.abort()
  }, [])


  // Part 3 - GN-level Wetland Scenario Coverage
  useEffect(() => {
    const controller = new AbortController()

    async function loadFutureGnData() {
      try {
        const result = await wetlandApi.futureGnSummary(
          controller.signal
        )

        if (!Array.isArray(result.features)) {
          throw new Error('Unexpected Part 3 GN data format')
        }

        setFutureGnCount(result.features.length)
      } catch (err) {
        if (err.name !== 'AbortError') {
          setFutureError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setFutureLoading(false)
        }
      }
    }

    loadFutureGnData()

    return () => controller.abort()
  }, [])


  // Part 4 - High and Very High Waterlogging Susceptibility
  useEffect(() => {
    const controller = new AbortController()

    async function loadWaterloggingData() {
      try {
        const result = await wetlandApi.waterloggingGnSummary(
          controller.signal
        )

        if (!Array.isArray(result.features) || result.features.length !== 56) {
          throw new Error('Incomplete or unexpected Part 4 GN data')
        }

        const properties = result.features.map(
          (feature) => feature.properties
        )

        const totalHighArea = properties.reduce(
          (sum, gn) =>
            sum + Number(gn.high_very_high_area_ha),
          0
        )

        const totalGeneralArea = properties.reduce(
          (sum, gn) =>
            sum + Number(gn.general_area_ha),
          0
        )

        if (
          !Number.isFinite(totalHighArea) ||
          !Number.isFinite(totalGeneralArea) ||
          totalGeneralArea <= 0
        ) {
          throw new Error('Invalid Part 4 area statistics')
        }

        setHighSusceptibilityArea(totalHighArea)
        setHighSusceptibilityShare(
          (totalHighArea / totalGeneralArea) * 100
        )
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setWaterloggingError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setWaterloggingLoading(false)
        }
      }
    }

    loadWaterloggingData()

    return () => controller.abort()
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs font-bold uppercase tracking-[0.2em] text-teal-700">
          Wetland Research Dashboard
        </p>

        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
          Kaduwela Wetland Intelligence
        </h1>

        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
          Explore urban wetland changes, candidate wetland areas,
          exploratory future scenarios and relative seasonal
          waterlogging susceptibility across Kaduwela.
        </p>
      </div>

      <div className="rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-teal-50 p-3 text-teal-700">
            <Droplets size={24} />
          </div>

          <div>
            <h2 className="text-lg font-bold text-slate-900">
              Wetland Research Overview
            </h2>
            <p className="text-xs text-slate-500">
              Four connected research components
            </p>
          </div>
        </div>
  
        <div className="mt-6 grid gap-4 md:grid-cols-2">

            {/* Part 1 - Monitored Wetlands */}
            <div className="group relative flex min-h-[250px] flex-col justify-between overflow-hidden rounded-[24px] bg-[#102E29] p-6 shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-xl">

            {/* Background Image */}
            <div
                className="absolute inset-0 bg-cover bg-center transition-transform duration-500 group-hover:scale-105"
                style={{
                backgroundImage: 'url("/images/wetland/wetland7.jpg")',
                }}
            />

            {/* Dark Gradient Overlay */}
            <div
                className="absolute inset-0"
                style={{
                background:
                    'linear-gradient(180deg, rgba(4,28,27,0.42) 0%, rgba(4,28,27,0.70) 48%, rgba(3,23,25,0.94) 100%)',
                }}
            />

            {/* Top Row */}
            <div className="relative z-10 flex items-start justify-between">
                <div className="rounded-xl border border-white/20 bg-white/15 p-3 text-white backdrop-blur-sm">
                <History size={22} />
                </div>

                <span className="rounded-full border border-white/20 bg-white/20 px-3 py-1 text-xs font-semibold text-white backdrop-blur-sm">
                Part 1
                </span>
            </div>

            {/* Live Statistic */}
            <div className="relative z-10 mt-8">
                <p className="text-xs font-bold uppercase tracking-[0.12em] text-teal-200">
                Monitored Wetlands
                </p>

                {loading ? (
                <p className="mt-3 text-sm text-white/80">
                    Loading...
                </p>
                ) : error ? (
                <p className="mt-3 text-sm text-red-200">
                    {error}
                </p>
                ) : (
                <p className="mt-2 text-5xl font-bold tracking-tight text-white">
                    {wetlandCount}
                </p>
                )}

                <div className="mt-3 h-1 w-12 rounded-full bg-teal-300" />

                <p className="mt-4 text-sm leading-6 text-white/90">
                Historical wetland monitoring locations recorded
                in our research dataset.
                </p>
            </div>
        </div>
        
            {/* Part 2 - Primary Wetland Candidates */}
            <div className="group relative flex min-h-[250px] flex-col justify-between overflow-hidden rounded-[24px] bg-[#102E29] p-6 shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-xl">

            {/* Background Image */}
            <div
                className="absolute inset-0 bg-cover bg-center transition-transform duration-500 group-hover:scale-105"
                style={{
                backgroundImage: 'url("/images/wetland/wetland6.jpg")',
                }}
            />

            {/* Dark Gradient Overlay */}
            <div
                className="absolute inset-0"
                style={{
                background:
                    'linear-gradient(180deg, rgba(4,28,27,0.42) 0%, rgba(4,28,27,0.70) 48%, rgba(3,23,25,0.94) 100%)',
                }}
            />

            {/* Top Row */}
            <div className="relative z-10 flex items-start justify-between">
                <div className="rounded-xl border border-white/20 bg-white/15 p-3 text-white backdrop-blur-sm">
                <Search size={22} />
                </div>

                <span className="rounded-full border border-white/20 bg-white/20 px-3 py-1 text-xs font-semibold text-white backdrop-blur-sm">
                Part 2
                </span>
            </div>

            {/* Live Statistic */}
            <div className="relative z-10 mt-8">
                <p className="text-xs font-bold uppercase tracking-[0.12em] text-teal-200">
                Primary Wetland Candidates
                </p>

                {candidateLoading ? (
                <p className="mt-3 text-sm text-white/80">
                    Loading...
                </p>
                ) : candidateError ? (
                <p className="mt-3 text-sm text-red-200">
                    {candidateError}
                </p>
                ) : (
                <p className="mt-2 text-5xl font-bold tracking-tight text-white">
                    {candidateCount}
                </p>
                )}

                <div className="mt-3 h-1 w-12 rounded-full bg-teal-300" />

                <p className="mt-4 text-sm leading-6 text-white/90">
                Screened candidate wetland polygons identified by
                our research. These are not independently confirmed wetlands.
                </p>
            </div>
            </div>

            {/* Part 3 - GN Divisions and Wetland Scenarios */}
            <div className="group relative flex min-h-[250px] flex-col justify-between overflow-hidden rounded-[24px] bg-[#102E29] p-6 shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-xl">

            {/* Background Image */}
            <div
                className="absolute inset-0 bg-cover bg-center transition-transform duration-500 group-hover:scale-105"
                style={{
                backgroundImage: 'url("/images/wetland/wetland5.jpg")',
                }}
            />

            {/* Dark Gradient Overlay */}
            <div
                className="absolute inset-0"
                style={{
                background:
                    'linear-gradient(180deg, rgba(4,28,27,0.42) 0%, rgba(4,28,27,0.70) 48%, rgba(3,23,25,0.94) 100%)',
                }}
            />

            {/* Top Row */}
            <div className="relative z-10 flex items-start justify-between">
                <div className="rounded-xl border border-white/20 bg-white/15 p-3 text-white backdrop-blur-sm">
                <CalendarRange size={22} />
                </div>

                <span className="rounded-full border border-white/20 bg-white/20 px-3 py-1 text-xs font-semibold text-white backdrop-blur-sm">
                Part 3
                </span>
            </div>

            {/* Live Statistic */}
            <div className="relative z-10 mt-8">
                <p className="text-xs font-bold uppercase tracking-[0.12em] text-teal-200">
                GN Divisions — Wetland Scenarios
                </p>

                {futureLoading ? (
                <p className="mt-3 text-sm text-white/80">
                    Loading...
                </p>
                ) : futureError ? (
                <p className="mt-3 text-sm text-red-200">
                    {futureError}
                </p>
                ) : (
                <p className="mt-2 text-5xl font-bold tracking-tight text-white">
                    {futureGnCount}
                </p>
                )}

                <div className="mt-3 h-1 w-12 rounded-full bg-teal-300" />

                <p className="mt-4 text-sm leading-6 text-white/90">
                GN divisions covered by historical wetland analysis
                and the exploratory 2030 wetland scenario.
                </p>
            </div>
            </div>

            {/* Part 4 - Waterlogging Susceptibility */}
            <div className="group relative flex min-h-[250px] flex-col justify-between overflow-hidden rounded-[24px] bg-[#102E29] p-6 shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-xl">

            {/* Background Image */}
            <div
                className="absolute inset-0 bg-cover bg-center transition-transform duration-500 group-hover:scale-105"
                style={{
                backgroundImage: 'url("/images/wetland/waterlogging2.jpg")',
                }}
            />

            {/* Dark Gradient Overlay */}
            <div
                className="absolute inset-0"
                style={{
                background:
                    'linear-gradient(180deg, rgba(4,28,27,0.42) 0%, rgba(4,28,27,0.70) 48%, rgba(3,23,25,0.94) 100%)',
                }}
            />

            {/* Top Row */}
            <div className="relative z-10 flex items-start justify-between">
                <div className="rounded-xl border border-white/20 bg-white/15 p-3 text-white backdrop-blur-sm">
                <Waves size={22} />
                </div>

                <span className="rounded-full border border-white/20 bg-white/20 px-3 py-1 text-xs font-semibold text-white backdrop-blur-sm">
                Part 4
                </span>
            </div>

            {/* Live Statistic */}
            <div className="relative z-10 mt-8">
                <p className="text-xs font-bold uppercase tracking-[0.12em] text-teal-200">
                High + Very High Susceptibility
                </p>

                {waterloggingLoading ? (
                <p className="mt-3 text-sm text-white/80">
                    Loading...
                </p>
                ) : waterloggingError ? (
                <p className="mt-3 text-sm text-red-200">
                    {waterloggingError}
                </p>
                ) : (
                <>
                    <p className="mt-2 text-5xl font-bold tracking-tight text-white">
                    {highSusceptibilityShare.toFixed(2)}%
                    </p>

                    <p className="mt-2 text-xs font-medium text-white/90">
                    {highSusceptibilityArea.toLocaleString('en-US', {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                    })} ha classified High or Very High
                    </p>
                </>
                )}

                <div className="mt-3 h-1 w-12 rounded-full bg-teal-300" />

                <p className="mt-4 text-sm leading-6 text-white/90">
                Share of assessed general-land area across 56 GN divisions.
                Relative susceptibility, not flood probability.
                </p>
            </div>
            </div>

        </div>

    </div>


    {/* Explore Research Components - Section Header */}
    <div className="mt-10 mb-5">
    <div className="flex items-start gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-teal-50 text-teal-700">
        <Droplets size={24} />
        </div>

        <div>
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-700">
            Explore Our Research
        </p>

        <h2 className="mt-1 text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
            Explore Research Components
        </h2>

        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
            Discover how historical wetland changes, candidate wetland
            screening, exploratory future scenarios and seasonal waterlogging
            susceptibility support environmental understanding and urban
            planning decisions in Kaduwela.
        </p>
        </div>
    </div>
    </div>


    {/* Research Component Image Cards */}
    <div className="grid gap-5 md:grid-cols-2">
    {researchParts.map(
        ({ title, description, icon: Icon, part, image }) => (
        <div
            key={part}
            className="group relative flex min-h-[270px] flex-col justify-between overflow-hidden rounded-[24px] border border-white/10 bg-slate-800 p-6 shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl md:p-7"
        >
            {/* Background Image */}
            <div
            className="absolute inset-0 bg-cover bg-center transition-transform duration-500 group-hover:scale-105"
            style={{
                backgroundImage: `url("${image}")`,
            }}
            />

            {/* Dark Gradient for Text Readability */}
            <div
            className="absolute inset-0"
            style={{
                background:
                'linear-gradient(180deg, rgba(5,30,30,0.30) 0%, rgba(5,30,30,0.60) 45%, rgba(4,25,28,0.94) 100%)',
            }}
            />

            {/* Top Row */}
            <div className="relative z-10 flex items-start justify-between">
            <div className="rounded-xl border border-white/20 bg-white/15 p-3 text-white backdrop-blur-sm">
                <Icon size={24} />
            </div>

            <span className="rounded-full border border-white/20 bg-white/20 px-3 py-1 text-xs font-semibold text-white backdrop-blur-sm">
                {part}
            </span>
            </div>

            {/* Card Content */}
            <div className="relative z-10 mt-10">
            <div className="mb-3 h-1 w-12 rounded-full bg-teal-300" />

            <h3 className="text-xl font-bold tracking-tight text-white md:text-2xl">
                {title}
            </h3>

            <p className="mt-3 max-w-md text-sm leading-6 text-white/85">
                {description}
            </p>

            <div className="mt-5 flex items-center gap-2 text-xs font-semibold text-teal-200">
                <span>Research Component</span>
                <span aria-hidden="true">→</span>
            </div>
            </div>
        </div>
        )
    )}
    </div>


      <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-900">
        Research interpretation: Candidate wetlands are screening
        outputs, the 2030 scenario is exploratory, and waterlogging
        susceptibility classes are not calibrated flood probabilities.
      </div>
    </div>
  )
}
