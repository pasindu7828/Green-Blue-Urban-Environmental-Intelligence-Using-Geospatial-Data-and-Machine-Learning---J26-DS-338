import {
  createContext,
  useContext,
  useMemo,
  useState,
} from 'react'

const UrbanGreenContext = createContext(null)

export function UrbanGreenProvider({ children }) {
  const [selectedGridId, setSelectedGridId] = useState(null)
  const [selectedGnId, setSelectedGnId] = useState(null)
  const [historicalYear, setHistoricalYear] = useState(2024)

  const selectGn = (gnId) => {
    setSelectedGnId(gnId || null)
    setSelectedGridId(null)
  }

  const value = useMemo(
    () => ({
      selectedGridId,
      setSelectedGridId,
      selectedGnId,
      setSelectedGnId,
      selectGn,
      historicalYear,
      setHistoricalYear,
    }),
    [selectedGridId, selectedGnId, historicalYear]
  )

  return (
    <UrbanGreenContext.Provider value={value}>
      {children}
    </UrbanGreenContext.Provider>
  )
}

export function useUrbanGreen() {
  const context = useContext(UrbanGreenContext)

  if (!context) {
    throw new Error(
      'useUrbanGreen must be used inside UrbanGreenProvider'
    )
  }

  return context
}
