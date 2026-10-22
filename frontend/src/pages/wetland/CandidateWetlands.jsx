import {
  Search,
  MapPinned,
  ChartNoAxesCombined,
  ShieldAlert,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { wetlandApi } from '../../features/wetland/services/wetlandApi'
import CandidateWetlandMap from '../../features/wetland/components/CandidateWetlandMap'


export default function CandidateWetlands() {

  const [primaryCandidates, setPrimaryCandidates] = useState(null)
  const [candidatesLoading, setCandidatesLoading] = useState(true)
  const [candidatesError, setCandidatesError] = useState(null)
  const [gnData, setGnData] = useState(null)
  const [gnLoading, setGnLoading] = useState(true)
  const [gnError, setGnError] = useState(null)
  const [candidateSearch, setCandidateSearch] = useState('')
  const [selectedPriority, setSelectedPriority] = useState('all')
  const [selectedQaAction, setSelectedQaAction] = useState('all')
  const [selectedCandidateId, setSelectedCandidateId] = useState(null)
  const [focusedCandidateId, setFocusedCandidateId] = useState(null)
  const candidateMapRef = useRef(null)
  const [gnSearch, setGnSearch] = useState('')
  const [gnCandidateFilter, setGnCandidateFilter] = useState('all')
  const [focusedGnId, setFocusedGnId] = useState(null)


  // Load Part 2 primary screened candidate polygons
  useEffect(() => {
    const controller = new AbortController()

    async function loadPrimaryCandidates() {
      try {
        const result = await wetlandApi.primaryCandidates(
          controller.signal
        )

        if (
          result.type !== 'FeatureCollection' ||
          !Array.isArray(result.features)
        ) {
          throw new Error('Unexpected primary candidate data format')
        }

        if (!controller.signal.aborted) {
          setPrimaryCandidates(result)
          setCandidatesError(null)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setCandidatesError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setCandidatesLoading(false)
        }
      }
    }

    loadPrimaryCandidates()

    return () => controller.abort()
  }, [])


  // Load Part 2 GN division candidate summaries
  useEffect(() => {
    const controller = new AbortController()

    async function loadGnData() {
      try {
        const result = await wetlandApi.candidateGnSummary(
          controller.signal
        )

        if (
          result.type !== 'FeatureCollection' ||
          !Array.isArray(result.features)
        ) {
          throw new Error('Unexpected GN division data format')
        }

        if (!controller.signal.aborted) {
          setGnData(result)
          setGnError(null)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setGnError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setGnLoading(false)
        }
      }
    }

    loadGnData()

    return () => controller.abort()
  }, [])



  // Available filters from actual research records
  const candidateFeatures = primaryCandidates?.features || []

  const availablePriorities = [
    ...new Set(
      candidateFeatures
        .map((feature) => feature.properties?.screening_priority)
        .filter(Boolean)
    ),
  ].sort()

  const availableQaActions = [
    ...new Set(
      candidateFeatures
        .map((feature) => feature.properties?.qa_final_action)
        .filter(Boolean)
    ),
  ].sort()

  // Search and filter primary screened candidates
  const filteredCandidates = [...candidateFeatures]
    .filter((feature) => {
      const candidate = feature.properties || {}
      const query = candidateSearch.trim().toLowerCase()

      const matchesSearch = [
        candidate.candidate_id,
        candidate.screening_priority,
        candidate.qa_final_action,
        candidate.qa_visual_confidence,
      ].some((value) =>
        String(value || '').toLowerCase().includes(query)
      )

      const matchesPriority =
        selectedPriority === 'all' ||
        candidate.screening_priority === selectedPriority

      const matchesQaAction =
        selectedQaAction === 'all' ||
        candidate.qa_final_action === selectedQaAction

      return matchesSearch && matchesPriority && matchesQaAction
    })
    .sort(
      (a, b) =>
        Number(a.properties?.qa_order ?? Infinity) -
        Number(b.properties?.qa_order ?? Infinity)
    )

    // Find the candidate selected by the user
    const selectedCandidate =
    candidateFeatures.find(
        (feature) =>
        feature.properties?.candidate_id === selectedCandidateId
    ) || null

const filteredGnDivisions = [...(gnData?.features || [])]
  .filter((feature) => {
    const gn = feature.properties || {}
    const query = gnSearch.trim().toLowerCase()

    const matchesSearch = [
      gn.gn_name,
      gn.gn_number,
    ].some((value) =>
      String(value ?? '').toLowerCase().includes(query)
    )

    const candidateCount = Number(
      gn.intersecting_primary_candidate_count ?? 0
    )

    const hasCandidates = candidateCount > 0

    const matchesCandidateFilter =
      gnCandidateFilter === 'all' ||
      (gnCandidateFilter === 'with' && hasCandidates) ||
      (gnCandidateFilter === 'without' && !hasCandidates)

    return matchesSearch && matchesCandidateFilter
  })
  .sort((a, b) =>
    String(a.properties?.gn_name || '').localeCompare(
      String(b.properties?.gn_name || '')
    )
  )

  return (
    <div className="space-y-6">

      {/* Page Header */}
      <div>
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-700">
          Wetland Research — Part 2
        </p>

        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
          Candidate Wetlands
        </h1>

        <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">
          Explore screened wetland candidate locations,
          their environmental indicators, manual review
          results and GN division-level screening summaries
          across the Kaduwela study area.
        </p>
      </div>

    {/* Candidate Summary */}
    <div className="grid gap-4 md:grid-cols-3">
      {[
        {
          title: 'Primary Screened Candidates',
          value: candidatesLoading
            ? '...'
            : candidatesError
              ? 'Unavailable'
              : candidateFeatures.length,
          description:
            'Primary screened candidate polygons, not independently confirmed wetlands.',
        },
        {
          title: 'GN Divisions Analysed',
          value: gnLoading
            ? '...'
            : gnError
              ? 'Unavailable'
              : gnData?.features?.length ?? 0,
          description:
            'GN divisions included in the candidate screening analysis.',
        },
        {
          title: 'High-Priority Candidates',
          value: candidatesLoading
            ? '...'
            : candidatesError
              ? 'Unavailable'
              : candidateFeatures.filter(
                  (feature) =>
                    feature.properties?.screening_priority === 'High'
                ).length,
          description:
            'Candidates classified as High screening priority; this is not a confirmed wetland classification.',
        },
      ].map((item) => (
        <div
          key={item.title}
          className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
        >
          <p className="text-xs font-semibold uppercase text-slate-500">
            {item.title}
          </p>

          <p className="mt-4 text-4xl font-bold text-teal-700">
            {item.value}
          </p>

          <p className="mt-2 text-xs leading-5 text-slate-500">
            {item.description}
          </p>
        </div>
      ))}
    </div>

      {/* Candidate Distribution Map */}
      <div
        ref={candidateMapRef}
        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
        >
        <div className="flex items-center gap-3">
          <MapPinned size={22} className="text-teal-700" />
          <h2 className="text-lg font-bold text-slate-900">
            Candidate Wetland Distribution Map
          </h2>
        </div>

        <p className="mt-2 text-sm text-slate-500">
          Explore primary screened candidate polygons
          with GN division boundaries.
        </p>

        {/* Interactive Candidate Wetland Distribution Map */}
        <div className="mt-5">

        {candidatesLoading || gnLoading ? (
            <div className="flex h-[430px] items-center justify-center rounded-xl bg-slate-50 text-sm text-slate-500">
            Loading candidate wetland geographic data...
            </div>
        ) : candidatesError ? (
            <div className="rounded-xl border border-red-200 bg-red-50 p-5 text-sm text-red-700">
            Unable to load candidate polygons: {candidatesError}
            </div>
        ) : (
            <>
            {gnError && (
                <p className="mb-3 text-sm text-amber-700">
                GN division boundaries could not be loaded: {gnError}
                </p>
            )}

          <CandidateWetlandMap
            candidateData={primaryCandidates}
            gnData={gnData}
            selectedCandidateId={focusedCandidateId}
            selectedGnId={focusedGnId}
          />


            </>
        )}

        </div>

      </div>

      {/* Candidate Records */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-3">
          <Search size={22} className="text-teal-700" />
          <h2 className="text-lg font-bold text-slate-900">
            Primary Candidate Records
          </h2>
        </div>

        <p className="mt-2 text-sm text-slate-500">
          Search candidate IDs, review screening priority,
          mapped area and available quality-assurance evidence.
        </p>


{/* Primary Candidate Records Table */}
<div className="mt-5">

{/* Candidate Search and Filters */}
<div className="mb-4 flex flex-wrap items-center gap-3">

  {/* Search Input */}
  <div className="relative min-w-[240px] flex-1">
    <Search
      size={18}
      className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
    />

    <input
      type="text"
      value={candidateSearch}
      onChange={(event) => setCandidateSearch(event.target.value)}
      placeholder="Search candidate ID, priority or QA action..."
      className="w-full rounded-xl border border-slate-200 bg-slate-50 py-3 pl-10 pr-4 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
    />
  </div>

  {/* Screening Priority Filter */}
  <select
    value={selectedPriority}
    onChange={(event) => setSelectedPriority(event.target.value)}
    aria-label="Filter by screening priority"
    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none focus:border-teal-500 sm:w-auto"
  >
    <option value="all">All Priorities</option>

    {availablePriorities.map((priority) => (
      <option key={priority} value={priority}>
        {priority}
      </option>
    ))}
  </select>

  {/* Manual QA Action Filter */}
  <select
    value={selectedQaAction}
    onChange={(event) => setSelectedQaAction(event.target.value)}
    aria-label="Filter by manual QA action"
    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none focus:border-teal-500 sm:w-auto"
  >
    <option value="all">All QA Actions</option>

    {availableQaActions.map((action) => (
      <option key={action} value={action}>
        {action.replaceAll('_', ' ')}
      </option>
    ))}
  </select>

  {/* Result Counter */}
  <p className="ml-auto text-xs font-medium text-slate-500">
    {candidatesLoading
      ? 'Loading records...'
      : `Showing ${filteredCandidates.length} of ${primaryCandidates?.features?.length ?? 0} candidates`}
  </p>

</div>


  {candidatesLoading ? (
    <p className="text-sm text-slate-500">
      Loading primary candidate records...
    </p>
  ) : candidatesError ? (
    <p className="text-sm text-red-600">
      Unable to load candidate records: {candidatesError}
    </p>
  ) : (
    <div className="overflow-x-auto rounded-xl border border-slate-200">

      <table className="w-full text-left text-sm">

        <thead className="bg-slate-50 text-xs uppercase text-slate-500">
          <tr>
            <th className="px-4 py-4">Candidate ID</th>
            <th className="px-4 py-4">Area (ha)</th>
            <th className="px-4 py-4">Screening Score</th>
            <th className="px-4 py-4">Priority</th>
            <th className="px-4 py-4">Manual QA Action</th>
            <th className="px-4 py-4">Visual Confidence</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-slate-100">

            {filteredCandidates.map((feature) => {

              const candidate = feature.properties || {}

              const area = Number(candidate.area_ha)
              const score = Number(candidate.mean_probability)

              return (
                <tr
                  key={candidate.candidate_id}
                  className="hover:bg-teal-50/40"
                >
                    <td className="px-4 py-4">
                    <button
                        type="button"
                        onClick={() =>
                        setSelectedCandidateId(candidate.candidate_id)
                        }
                        className={`font-bold transition hover:underline ${
                        selectedCandidateId === candidate.candidate_id
                            ? 'text-teal-900 underline'
                            : 'text-teal-700'
                        }`}
                    >
                        {candidate.candidate_id || 'N/A'}
                    </button>
                    </td>

                  <td className="px-4 py-4">
                    {candidate.area_ha == null || !Number.isFinite(area)
                      ? 'N/A'
                      : area.toFixed(2)}
                  </td>

                  <td className="px-4 py-4">
                    {candidate.mean_probability == null || !Number.isFinite(score)
                      ? 'N/A'
                      : score.toFixed(4)}
                  </td>

                  <td className="px-4 py-4">
                    {candidate.screening_priority || 'N/A'}
                  </td>

                  <td className="px-4 py-4">
                    {candidate.qa_final_action || 'N/A'}
                  </td>

                  <td className="px-4 py-4">
                    {candidate.qa_visual_confidence || 'N/A'}
                  </td>
                </tr>
              )
            })}

        </tbody>

      </table>

    </div>
  )}

{/* Selected Candidate Details Popup */}
{selectedCandidate && (
  <div
    className="fixed inset-0 z-[2000] flex items-center justify-center bg-slate-950/60 p-4"
    onClick={() => setSelectedCandidateId(null)}
  >
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="candidate-popup-title"
      className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl bg-white p-6 shadow-2xl"
      onClick={(event) => event.stopPropagation()}
    >

      {/* Popup Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-widest text-teal-700">
            Wetland Research — Part 2
          </p>

          <h3
            id="candidate-popup-title"
            className="mt-2 text-2xl font-bold text-slate-900"
          >
            Candidate {selectedCandidate.properties.candidate_id}
          </h3>

          <p className="mt-2 text-sm text-slate-500">
            Individual screening and manual review information
          </p>
        </div>

        <button
          type="button"
          onClick={() => setSelectedCandidateId(null)}
          aria-label="Close candidate details"
          className="rounded-xl bg-slate-100 px-3 py-2 text-lg font-bold text-slate-600 hover:bg-slate-200"
        >
          ×
        </button>
      </div>

      {/* Candidate Details */}
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

        {[
          {
            label: 'Candidate Area',
            value:
              selectedCandidate.properties.area_ha == null
                ? 'N/A'
                : `${Number(
                    selectedCandidate.properties.area_ha
                  ).toFixed(2)} ha`,
          },
          {
            label: 'Mean Screening Score',
            value:
              selectedCandidate.properties.mean_probability == null
                ? 'N/A'
                : Number(
                    selectedCandidate.properties.mean_probability
                  ).toFixed(4),
          },
          {
            label: 'Screening Priority',
            value:
              selectedCandidate.properties.screening_priority || 'N/A',
          },
          {
            label: 'Manual QA Action',
            value:
              selectedCandidate.properties.qa_final_action || 'N/A',
          },
          {
            label: 'Visual Confidence',
            value:
              selectedCandidate.properties.qa_visual_confidence || 'N/A',
          },
          {
            label: 'Visual Category',
            value:
              selectedCandidate.properties.qa_final_visual_category || 'N/A',
          },
        ].map((item) => (
          <div
            key={item.label}
            className="rounded-xl border border-slate-200 bg-slate-50 p-4"
          >
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              {item.label}
            </p>

            <p className="mt-2 break-words text-base font-bold text-teal-700">
              {item.value}
            </p>
          </div>
        ))}

      </div>

      {/* Research Warning */}
      <div className="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-4">
        <p className="text-xs leading-5 text-amber-900">
          This location is a screened wetland candidate,
          not an independently confirmed wetland.
          Screening scores are not calibrated probabilities.
          Manual visual review does not replace field verification.
        </p>
      </div>

      {/* Close Button */}

{/* Popup Actions */}
<div className="mt-6 flex flex-wrap justify-end gap-3">

  <button
    type="button"
    onClick={() => setSelectedCandidateId(null)}
    className="rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50"
  >
    Close Details
  </button>

  <button
    type="button"
    onClick={() => {
      setFocusedCandidateId(
        selectedCandidate.properties.candidate_id
      )

      setSelectedCandidateId(null)

      candidateMapRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
    }}
    className="rounded-xl bg-teal-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-teal-800"
  >
    View on Map
  </button>

</div>


    </div>
  </div>
)}

  <p className="mt-4 text-xs leading-5 text-slate-500">
    These records represent primary screened wetland candidates.
    Screening scores are model outputs, not calibrated probabilities
    of wetland presence. Manual visual review does not constitute
    independent field verification.
  </p>

</div>

      </div>

      {/* GN Division Summary */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-3">
          <ChartNoAxesCombined size={22} className="text-teal-700" />
          <h2 className="text-lg font-bold text-slate-900">
            GN Division Candidate Analysis
          </h2>
        </div>

        <p className="mt-2 text-sm text-slate-500">
          Review candidate intersections, candidate area
          and screening indicators across GN divisions.
        </p>


{/* GN Division Candidate Analysis Table */}
<div className="mt-5">

{/* GN Division Search */}
<div className="mb-4 flex flex-wrap items-center gap-4">
  <div className="relative min-w-[240px] flex-1">
    <Search
      size={18}
      className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
    />

    <input
      type="text"
      value={gnSearch}
      onChange={(event) => setGnSearch(event.target.value)}
      placeholder="Search GN division name or number..."
      className="w-full rounded-xl border border-slate-200 bg-slate-50 py-3 pl-10 pr-4 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
    />
  </div>
  {/* Candidate Presence Filter */}
<select
  value={gnCandidateFilter}
  onChange={(event) => setGnCandidateFilter(event.target.value)}
  aria-label="Filter GN divisions by candidate presence"
  className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none focus:border-teal-500 sm:w-auto"
>
  <option value="all">All GN Divisions</option>
  <option value="with">With Candidates</option>
  <option value="without">Without Candidates</option>
</select>

  <p className="text-xs font-medium text-slate-500">
    {gnLoading
      ? 'Loading GN divisions...'
      : `Showing ${filteredGnDivisions.length} of ${gnData?.features?.length ?? 0} GN divisions`}
  </p>
</div>

  {gnLoading ? (
    <p className="text-sm text-slate-500">
      Loading GN division screening data...
    </p>
  ) : gnError ? (
    <p className="text-sm text-red-600">
      Unable to load GN division data: {gnError}
    </p>
  ) : (
    <div className="overflow-x-auto rounded-xl border border-slate-200">

      <table className="w-full text-left text-sm">

        <thead className="bg-slate-50 text-xs uppercase text-slate-500">
          <tr>
            <th className="px-4 py-4">GN Division</th>
            <th className="px-4 py-4">GN Number</th>
            <th className="px-4 py-4">Intersecting Candidates</th>
            <th className="px-4 py-4">Candidate Area (ha)</th>
            <th className="px-4 py-4">GN Area Covered (%)</th>
            <th className="px-4 py-4">High Priority</th>
            <th className="px-4 py-4">Medium Priority</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-slate-100">
            {filteredGnDivisions.map((feature) => {
              const gn = feature.properties || {}

              const candidateArea = Number(
                gn.primary_candidate_area_ha
              )

              const candidatePercent = Number(
                gn.primary_candidate_area_pct_of_gn
              )

              return (
                <tr
                  key={gn.gn_id}
                  className="hover:bg-teal-50/40"
                >
                <td className="px-4 py-4">
                  <button
                    type="button"
                    onClick={() => {
                      setFocusedCandidateId(null)
                      setFocusedGnId(gn.gn_id)

                      candidateMapRef.current?.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start',
                      })
                    }}
                    className="font-semibold text-teal-700 transition hover:text-teal-900 hover:underline"
                    title={`View ${gn.gn_name || 'GN division'} on map`}
                  >
                    {gn.gn_name || 'N/A'}
                  </button>
                </td>


                  <td className="px-4 py-4">
                    {gn.gn_number || 'N/A'}
                  </td>

                  <td className="px-4 py-4">
                    {gn.intersecting_primary_candidate_count ?? 0}
                  </td>

                  <td className="px-4 py-4">
                    {gn.primary_candidate_area_ha == null ||
                    !Number.isFinite(candidateArea)
                      ? 'N/A'
                      : candidateArea.toFixed(2)}
                  </td>

                  <td className="px-4 py-4">
                    {gn.primary_candidate_area_pct_of_gn == null ||
                    !Number.isFinite(candidatePercent)
                      ? 'N/A'
                      : `${candidatePercent.toFixed(2)}%`}
                  </td>

                  <td className="px-4 py-4">
                    {gn.high_intersecting_count ?? 0}
                  </td>

                  <td className="px-4 py-4">
                    {gn.medium_intersecting_count ?? 0}
                  </td>
                </tr>
              )
            })}

        </tbody>
      </table>

    </div>
  )}

  <p className="mt-4 text-xs leading-5 text-slate-500">
    GN division results represent intersections with screened
    primary candidate polygons. A candidate may intersect
    multiple GN divisions, so intersecting candidate counts
    must not be added together to calculate the number of
    unique candidates.
  </p>

</div>

      </div>

      {/* Scientific Interpretation */}
      <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4">
        <ShieldAlert
          size={20}
          className="shrink-0 text-amber-700"
        />

        <p className="text-xs leading-5 text-amber-900">
          These results represent screened wetland candidates,
          not independently confirmed wetlands. Manual visual
          review and model screening scores do not replace
          field verification. Candidate screening scores must
          not be interpreted as calibrated certainty.
        </p>
      </div>

    </div>
  )
}
