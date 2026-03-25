import { useEffect, useMemo, useState } from "react"
import "./App.css"
import { checkPresence, fetchCollectionCount, imageUrl } from "./api"
import type { PresenceResponse } from "./api"

function scoreLabel(score: number | null | undefined): string {
  if (score === null || score === undefined) return "n/a"
  return score.toFixed(3)
}

function App() {
  const [collection, setCollection] = useState<string>("image_embeddings_512")
  const [count, setCount] = useState<number>(0)
  const [backendError, setBackendError] = useState<string>("")

  const [imageFile, setImageFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string>("")

  const [threshold, setThreshold] = useState<number>(0.85)
  const [topK, setTopK] = useState<number>(5)
  const [loading, setLoading] = useState<boolean>(false)

  const [result, setResult] = useState<PresenceResponse | null>(null)
  const [error, setError] = useState<string>("")

  useEffect(() => {
    let mounted = true
    fetchCollectionCount()
      .then((data) => {
        if (!mounted) return
        setCollection(data.collection)
        setCount(data.count)
      })
      .catch((e) => {
        if (!mounted) return
        setBackendError((e as any)?.message ?? String(e))
      })
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    if (!imageFile) return
    const url = URL.createObjectURL(imageFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [imageFile])

  const presentClass = useMemo(() => {
    if (!result) return ""
    return result.present ? "badge badgePresent" : "badge badgeMissing"
  }, [result])

  async function onCheck() {
    setError("")
    setBackendError("")
    setResult(null)

    if (!imageFile) {
      setError("Please upload a query image first.")
      return
    }

    try {
      setLoading(true)
      const res = await checkPresence({ imageFile, threshold, topK })
      setResult(res)
    } catch (e: any) {
      setError(e?.message ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Image presence check</h1>
          <p>Upload a query image and check whether a similar image already exists in Qdrant.</p>
        </div>
        <div className="meta">
          <div className="metaRow">
            <span className="metaKey">Collection</span>
            <span className="metaVal">{collection}</span>
          </div>
          <div className="metaRow">
            <span className="metaKey">Vectors</span>
            <span className="metaVal">{count}</span>
          </div>
        </div>
      </header>

      {(backendError || error) && (
        <div className="callout calloutError">
          <div className="calloutTitle">Problem</div>
          <div>{backendError || error}</div>
        </div>
      )}

      <div className="grid">
        <section className="card">
          <h2>Query</h2>
          <div className="uploadWrap">
            {previewUrl ? (
              <img className="preview" src={previewUrl} alt="Query preview" />
            ) : (
              <div className="dropHint">Choose an image file to upload</div>
            )}
            <label className="fileBtn">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const f = e.target.files?.[0] ?? null
                  setImageFile(f)
                  setResult(null)
                  setError("")
                }}
              />
              Upload
            </label>
          </div>

          <div className="controls">
            <label className="control">
              <div className="controlRow">
                <span>Threshold</span>
                <span className="mono">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </label>

            <label className="control">
              <div className="controlRow">
                <span>Top K</span>
                <span className="mono">{topK}</span>
              </div>
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value))))}
              />
            </label>

            <button className="primaryBtn" onClick={onCheck} disabled={loading}>
              {loading ? "Checking..." : "Check presence"}
            </button>
          </div>
        </section>

        <section className="card resultsCard">
          <h2>Result</h2>
          {result ? (
            <>
              <div className="resultHeader">
                <span className={presentClass}>{result.present ? "Present" : "Not present"}</span>
                <div className="resultLine">
                  Best similarity: <span className="mono">{scoreLabel(result.best_score)}</span>
                  <span className="dim"> / threshold </span>
                  <span className="mono">{result.threshold.toFixed(2)}</span>
                </div>
              </div>

              <div className="matches">
                {result.results.length === 0 ? (
                  <div className="empty">No results found in the collection.</div>
                ) : (
                  result.results.map((m) => (
                    <div key={m.id} className="match">
                      <div className="thumb">
                        {m.file_name ? (
                          <img src={imageUrl(m.file_name)} alt={m.file_name ?? "match"} />
                        ) : (
                          <div className="thumbPlaceholder">No file</div>
                        )}
                      </div>
                      <div className="matchInfo">
                        <div className="matchName">{m.file_name ?? "unknown"}</div>
                        <div className="matchScore">
                          Similarity: <span className="mono">{m.score.toFixed(3)}</span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </>
          ) : (
            <div className="empty">
              Upload a query image and click <span className="mono">Check presence</span>.
            </div>
          )}
        </section>
      </div>

      <footer className="footer">
        <div>
          Backend: <span className="mono">/api/*</span>. Images are served from your local <span className="mono">images/</span>{" "}
          folder.
        </div>
      </footer>
    </div>
  )
}

export default App
