import { useState, useEffect } from 'react'

export default function useFirstVisit(key = 'hasVisited') {
  const [isFirstVisit, setIsFirstVisit] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const hasVisited = localStorage.getItem(key)
    setIsFirstVisit(!hasVisited)
    setIsLoading(false)
  }, [key])

  const markAsVisited = () => {
    localStorage.setItem(key, 'true')
    setIsFirstVisit(false)
  }

  const reset = () => {
    localStorage.removeItem(key)
    setIsFirstVisit(true)
  }

  return { isFirstVisit, isLoading, markAsVisited, reset }
}
