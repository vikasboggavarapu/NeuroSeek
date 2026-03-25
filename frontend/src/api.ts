export type PresenceMatch = {
  id: string
  score: number
  file_name?: string | null
}

export type PresenceResponse = {
  present: boolean
  best_score: number | null
  threshold: number
  results: PresenceMatch[]
}

const API_ORIGIN = import.meta.env.VITE_API_ORIGIN ?? "http://localhost:8000"

export async function fetchCollectionCount(): Promise<{ collection: string; count: number }> {
  const res = await fetch(`${API_ORIGIN}/api/collection/count`)
  if (!res.ok) throw new Error(`Failed to fetch collection count (${res.status})`)
  return res.json()
}

export async function checkPresence(params: {
  imageFile: File
  threshold: number
  topK: number
}): Promise<PresenceResponse> {
  const form = new FormData()
  form.append("image", params.imageFile)

  const url = new URL(`${API_ORIGIN}/api/presence`)
  url.searchParams.set("threshold", String(params.threshold))
  url.searchParams.set("top_k", String(params.topK))

  const res = await fetch(url.toString(), {
    method: "POST",
    body: form,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    throw new Error(`Presence check failed (${res.status}): ${text || res.statusText}`)
  }
  return res.json()
}

export function imageUrl(fileName: string | null | undefined): string {
  if (!fileName) return ""
  const API = `${API_ORIGIN}/api/image`
  const url = new URL(API)
  url.searchParams.set("file_name", fileName)
  return url.toString()
}

