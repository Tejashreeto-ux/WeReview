import { useState, useEffect, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  Star,
  ShieldCheck,
  CheckCircle,
  AlertTriangle,
  Send,
  ChevronRight,
  Heart,
  ShoppingCart,
  Truck,
  RotateCcw,
  RefreshCw,
  User,
  Calendar,
} from 'lucide-react'
import { PRODUCTS } from '../data/products'
import { useAuth } from '../context/AuthContext'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:4000'

function StarRating({ value, onChange, readonly = false, size = 20 }) {
  const [hover, setHover] = useState(0)
  return (
    <div className="flex gap-1 items-center">
      {[1, 2, 3, 4, 5].map(star => (
        <button
          key={star}
          type="button"
          disabled={readonly}
          onClick={() => !readonly && onChange && onChange(star)}
          onMouseEnter={() => !readonly && setHover(star)}
          onMouseLeave={() => !readonly && setHover(0)}
          className={readonly ? 'cursor-default' : 'cursor-pointer hover:scale-110 transition-transform'}
        >
          <Star
            size={readonly ? (size || 13) : 22}
            className={
              star <= (hover || value)
                ? 'fill-yellow-400 text-yellow-400'
                : 'fill-gray-200 text-gray-200'
            }
          />
        </button>
      ))}
    </div>
  )
}

function formatDate(dateString) {
  if (!dateString) return 'Recent'
  try {
    const d = new Date(dateString)
    if (isNaN(d.getTime())) return 'Recent'
    return d.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  } catch {
    return 'Recent'
  }
}

export default function ProductDetails() {
  const { user } = useAuth()
  const { id } = useParams()
  const product = PRODUCTS.find(p => p.id === Number(id) || p.asin === id) || PRODUCTS[0]
  const productAsin = product.asin || String(product.id)

  // Reviews from backend
  const [reviews, setReviews] = useState([])
  const [loadingReviews, setLoadingReviews] = useState(true)
  const [fetchError, setFetchError] = useState(null)

  // Review submission state
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const [rating, setRating] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [submitResult, setSubmitResult] = useState(null)
  const [submitError, setSubmitError] = useState('')

  // Fetch reviews from Node.js backend filtered by ASIN
  const fetchReviews = useCallback(async () => {
    setLoadingReviews(true)
    setFetchError(null)
    try {
      const res = await fetch(`${API_URL}/api/reviews?asin=${encodeURIComponent(productAsin)}`)
      if (!res.ok) {
        throw new Error(`Failed to load reviews (HTTP ${res.status})`)
      }
      const data = await res.json()
      if (data.status === 'success' && Array.isArray(data.reviews)) {
        setReviews(data.reviews)
      } else {
        setReviews([])
      }
    } catch (err) {
      console.error('Error fetching reviews:', err)
      setFetchError(err.message || 'Unable to load reviews from server.')
    } finally {
      setLoadingReviews(false)
    }
  }, [productAsin])

  useEffect(() => {
    fetchReviews()
  }, [fetchReviews])

  // Filter backend reviews into Genuine vs Potentially Fake
  const genuineReviews = reviews.filter(
    r => (r.prediction || '').toLowerCase() === 'genuine'
  )
  const fakeReviews = reviews.filter(
    r => (r.prediction || '').toLowerCase() === 'fake'
  )

  // Handle Review Submission
  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!user) {
      setSubmitError('Please sign in or create an account to submit a review.')
      return
    }

    if (!title.trim() || !text.trim()) {
      setSubmitError('Review title and text are required.')
      return
    }
    if (rating === 0) {
      setSubmitError('Please select a star rating (1–5 stars).')
      return
    }

    setSubmitError('')
    setSubmitting(true)
    setSubmitResult(null)

    try {
      const payload = {
        asin: productAsin,
        review_title: title.trim(),
        review_text: text.trim(),
        rating: Number(rating),
      }

      console.log('[1. Frontend -> Node.js] Sending review payload to /api/reviews/predict:');
      console.log(payload);

      const res = await fetch(`${API_URL}/api/reviews/predict`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      console.log('[2. Node.js -> Frontend] Received prediction response:');
      console.log(data);

      if (!res.ok) {
        throw new Error(data.message || data.error || `Submission failed with status ${res.status}`);
      }

      // Display actual prediction and fake probability
      setSubmitResult(data);

      // Clear form inputs
      setTitle('')
      setText('')
      setRating(0)

      // Refresh reviews list automatically without reloading page
      await fetchReviews()
    } catch (err) {
      console.error('Error submitting review:', err)
      setSubmitError(err.message || 'Failed to submit review to backend server.')
    } finally {
      setSubmitting(false)
    }
  }

  const discount = product.originalPrice
    ? Math.round(((product.originalPrice - product.price) / product.originalPrice) * 100)
    : null

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-1 text-xs text-gray-400 mb-6">
        <Link to="/" className="hover:text-purple-600">Home</Link>
        <ChevronRight size={12} />
        <Link to="/shop" className="hover:text-purple-600">{product.category}</Link>
        <ChevronRight size={12} />
        <span className="text-gray-700">{product.name}</span>
        <span className="ml-2 text-gray-400 font-mono text-[11px] bg-gray-100 px-2 py-0.5 rounded">
          ASIN: {productAsin}
        </span>
      </nav>

      {/* Product Info */}
      <div className="grid md:grid-cols-2 gap-10 mb-12">
        {/* Product Image */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex items-center justify-center">
          <img
            src={product.image}
            alt={product.name}
            className="w-72 h-72 object-cover rounded-xl"
            onError={e => { e.target.src = 'https://placehold.co/300x300?text=Product' }}
          />
        </div>

        {/* Product Details */}
        <div>
          <span className="text-xs uppercase tracking-wide text-purple-600 font-semibold">{product.category}</span>
          <h1 className="text-2xl font-bold text-gray-900 mt-1 mb-2">{product.name}</h1>
          <div className="flex items-center gap-2 mb-3">
            <StarRating value={Math.round(product.rating)} readonly size={16} />
            <span className="text-sm text-gray-600 font-medium">
              {product.rating} ({reviews.length > 0 ? reviews.length : product.reviews} reviews)
            </span>
          </div>
          <div className="flex items-baseline gap-3 mb-4">
            <span className="text-3xl font-bold text-gray-900">₹{product.price.toLocaleString()}</span>
            {product.originalPrice && (
              <span className="text-lg text-gray-400 line-through">₹{product.originalPrice.toLocaleString()}</span>
            )}
            {discount && (
              <span className="bg-green-100 text-green-700 text-sm font-bold px-2 py-0.5 rounded">
                {discount}% OFF
              </span>
            )}
          </div>
          <div className="space-y-2 mb-6 text-sm text-gray-600">
            {['High quality product & verified materials', 'Manufacturer warranty included', 'Genuine customer feedback protection', 'Fast order processing'].map(f => (
              <div key={f} className="flex items-center gap-2">
                <CheckCircle size={14} className="text-green-500 shrink-0" />
                <span>{f}</span>
              </div>
            ))}
          </div>
          <div className="flex gap-3 mb-6">
            <button className="flex-1 flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 rounded-xl transition-colors">
              <ShoppingCart size={18} /> Add to Cart
            </button>
            <button className="p-3 border border-gray-200 rounded-xl hover:border-red-300 hover:text-red-500 transition-colors">
              <Heart size={18} />
            </button>
          </div>
          <div className="flex gap-4 text-sm text-gray-500">
            <div className="flex items-center gap-1.5"><Truck size={15} className="text-purple-500" /> Free Delivery</div>
            <div className="flex items-center gap-1.5"><RotateCcw size={15} className="text-purple-500" /> 30-Day Returns</div>
          </div>
        </div>
      </div>

      {/* AI Analysis Banner */}
      <div className="bg-purple-50 border border-purple-100 rounded-xl p-4 mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <ShieldCheck size={24} className="text-purple-600 shrink-0" />
          <div>
            <div className="font-semibold text-purple-900">AI Review Intelligence Active</div>
            <div className="text-sm text-purple-700">
              Reviews for ASIN <code className="font-mono bg-purple-100 px-1.5 py-0.5 rounded text-purple-800">{productAsin}</code> are analyzed by our trained machine learning model.
            </div>
          </div>
        </div>
        <button
          onClick={fetchReviews}
          disabled={loadingReviews}
          className="flex items-center gap-1.5 text-xs text-purple-700 font-semibold bg-white border border-purple-200 px-3 py-1.5 rounded-lg hover:bg-purple-100 transition-colors disabled:opacity-60"
        >
          <RefreshCw size={13} className={loadingReviews ? 'animate-spin' : ''} />
          {loadingReviews ? 'Refreshing...' : 'Refresh Reviews'}
        </button>
      </div>

      {/* Fetch Error Alert */}
      {fetchError && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mb-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle size={18} className="shrink-0" />
            <span className="text-sm">{fetchError}</span>
          </div>
          <button
            onClick={fetchReviews}
            className="text-xs bg-red-100 hover:bg-red-200 text-red-800 font-semibold px-3 py-1 rounded"
          >
            Retry
          </button>
        </div>
      )}

      {/* Reviews Grid: Genuine vs Potentially Fake */}
      <div className="grid md:grid-cols-2 gap-6 mb-10">
        {/* Genuine Reviews Section */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <div className="flex items-center justify-between mb-4 border-b border-gray-100 pb-3">
            <h2 className="font-bold text-green-700 flex items-center gap-2 text-base">
              <CheckCircle size={18} className="text-green-500" />
              Genuine Reviews
            </h2>
            <span className="text-xs font-bold bg-green-100 text-green-800 px-2.5 py-0.5 rounded-full">
              {genuineReviews.length}
            </span>
          </div>

          {loadingReviews ? (
            <div className="text-center py-10 text-gray-400 text-sm">
              <RefreshCw size={20} className="animate-spin mx-auto mb-2 text-purple-500" />
              Loading genuine reviews...
            </div>
          ) : genuineReviews.length === 0 ? (
            <div className="text-center py-10 text-gray-400 text-sm">
              No genuine reviews submitted yet for this product.
            </div>
          ) : (
            <div className="space-y-3">
              {genuineReviews.map(r => {
                const fakeProbPercent =
                  r.fake_probability !== null && r.fake_probability !== undefined
                    ? (Number(r.fake_probability) * 100).toFixed(1)
                    : null
                return (
                  <div key={r.id} className="bg-green-50/40 border border-green-100 rounded-xl p-4 shadow-2xs">
                    <div className="flex items-center justify-between mb-1.5">
                      <h4 className="font-semibold text-gray-900 text-sm">{r.review_title || 'Untitled Review'}</h4>
                      {/* Actual Prediction & Fake Probability from Backend */}
                      <span className="text-[11px] font-semibold bg-green-100 text-green-800 px-2 py-0.5 rounded-full flex items-center gap-1">
                        <CheckCircle size={11} />
                        {r.prediction ? r.prediction.toUpperCase() : 'GENUINE'}
                        {fakeProbPercent !== null && ` (${fakeProbPercent}% fake risk)`}
                      </span>
                    </div>

                    <div className="mb-2">
                      <StarRating value={Math.round(r.review_rating || 5)} readonly size={12} />
                    </div>

                    <p className="text-sm text-gray-700 mb-3 whitespace-pre-wrap">{r.review_text}</p>

                    <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-green-100/60">
                      <div className="flex items-center gap-1.5">
                        <div className="w-5 h-5 rounded-full bg-green-200 flex items-center justify-center text-green-800 font-bold text-[10px]">
                          {(r.reviewer_id || 'U')[0].toUpperCase()}
                        </div>
                        <span className="font-medium text-gray-700">{r.reviewer_id || 'Anonymous'}</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-400">
                        <Calendar size={12} />
                        <span>{formatDate(r.created_at || r.review_date)}</span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Potentially Fake Reviews Section */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <div className="flex items-center justify-between mb-4 border-b border-gray-100 pb-3">
            <h2 className="font-bold text-red-700 flex items-center gap-2 text-base">
              <AlertTriangle size={18} className="text-red-500" />
              Potentially Fake Reviews
            </h2>
            <span className="text-xs font-bold bg-red-100 text-red-800 px-2.5 py-0.5 rounded-full">
              {fakeReviews.length}
            </span>
          </div>

          {loadingReviews ? (
            <div className="text-center py-10 text-gray-400 text-sm">
              <RefreshCw size={20} className="animate-spin mx-auto mb-2 text-purple-500" />
              Loading fake reviews...
            </div>
          ) : fakeReviews.length === 0 ? (
            <div className="text-center py-10 text-gray-400 text-sm">
              No potentially fake reviews detected for this product.
            </div>
          ) : (
            <div className="space-y-3">
              {fakeReviews.map(r => {
                const fakeProbPercent =
                  r.fake_probability !== null && r.fake_probability !== undefined
                    ? (Number(r.fake_probability) * 100).toFixed(1)
                    : null
                return (
                  <div key={r.id} className="bg-red-50/40 border border-red-100 rounded-xl p-4 shadow-2xs">
                    <div className="flex items-center justify-between mb-1.5">
                      <h4 className="font-semibold text-gray-900 text-sm">{r.review_title || 'Untitled Review'}</h4>
                      {/* Actual Prediction & Fake Probability from Backend - No Hardcoded Labels */}
                      <span className="text-[11px] font-semibold bg-red-100 text-red-800 px-2 py-0.5 rounded-full flex items-center gap-1">
                        <AlertTriangle size={11} />
                        {r.prediction ? r.prediction.toUpperCase() : 'FAKE'}
                        {fakeProbPercent !== null && ` (${fakeProbPercent}% fake probability)`}
                      </span>
                    </div>

                    <div className="mb-2">
                      <StarRating value={Math.round(r.review_rating || 1)} readonly size={12} />
                    </div>

                    <p className="text-sm text-gray-700 mb-3 whitespace-pre-wrap">{r.review_text}</p>

                    <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-red-100/60">
                      <div className="flex items-center gap-1.5">
                        <div className="w-5 h-5 rounded-full bg-red-200 flex items-center justify-center text-red-800 font-bold text-[10px]">
                          {(r.reviewer_id || 'U')[0].toUpperCase()}
                        </div>
                        <span className="font-medium text-gray-700">{r.reviewer_id || 'Anonymous'}</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-400">
                        <Calendar size={12} />
                        <span>{formatDate(r.created_at || r.review_date)}</span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Review Submission Form */}
      <div className="bg-white border border-gray-100 rounded-2xl shadow-sm p-6">
        <h2 className="text-lg font-bold text-gray-900 mb-1 flex items-center gap-2">
          <Send size={18} className="text-purple-600" /> Write Your Review
        </h2>
        <p className="text-sm text-gray-500 mb-5">
          Your review will be submitted to the AI backend, evaluated with the trained ML model, and stored in MySQL for ASIN{' '}
          <strong className="font-mono text-purple-700">{productAsin}</strong>.
        </p>

        {/* Prediction Result Card */}
        {submitResult && (() => {
          const isGenuine = String(submitResult.prediction || '').toLowerCase() === 'genuine'
          const fakePct = (Number(submitResult.fake_probability) * 100).toFixed(1)
          const genuinePct = (Number(submitResult.genuine_probability) * 100).toFixed(1)
          const textPct = submitResult.details ? (Number(submitResult.details.text_fake_probability) * 100).toFixed(1) : null
          const behaviorPct = submitResult.details ? (Number(submitResult.details.behavior_fake_probability) * 100).toFixed(1) : null

          return (
            <div className={`rounded-2xl border-2 mb-5 overflow-hidden ${isGenuine ? 'border-green-300' : 'border-red-300'}`}>
              {/* Verdict Banner */}
              <div className={`flex items-center justify-between px-5 py-4 ${isGenuine ? 'bg-green-500' : 'bg-red-500'}`}>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white text-xl">
                    {isGenuine ? '✅' : '⚠️'}
                  </div>
                  <div>
                    <div className="text-white font-extrabold text-lg tracking-wide">
                      {isGenuine ? 'GENUINE REVIEW' : 'FAKE REVIEW'}
                    </div>
                    <div className="text-white/80 text-xs font-medium">
                      AI Verdict — We-Review ML Model
                    </div>
                  </div>
                </div>
                <div className={`text-right px-4 py-2 rounded-xl ${isGenuine ? 'bg-green-600' : 'bg-red-600'}`}>
                  <div className="text-white text-2xl font-black">
                    {isGenuine ? genuinePct : fakePct}%
                  </div>
                  <div className="text-white/80 text-[10px] font-semibold uppercase tracking-wider">
                    {isGenuine ? 'Genuine' : 'Fake'} Confidence
                  </div>
                </div>
              </div>

              {/* Probability Bar */}
              <div className="px-5 py-3 bg-gray-50 border-b border-gray-100">
                <div className="flex justify-between text-[11px] font-semibold mb-1">
                  <span className="text-green-700">Genuine {genuinePct}%</span>
                  <span className="text-red-600">Fake {fakePct}%</span>
                </div>
                <div className="h-2.5 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-400 to-green-500 rounded-full transition-all"
                    style={{ width: `${genuinePct}%` }}
                  />
                </div>
              </div>

              {/* Details Row */}
              <div className="px-5 py-3 bg-white flex flex-wrap items-center gap-4 text-xs text-gray-600">
                {textPct && (
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-purple-400 inline-block"/>
                    Text Model: <strong className="text-gray-800">{textPct}% fake risk</strong>
                  </span>
                )}
                {behaviorPct && (
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-indigo-400 inline-block"/>
                    Behavior Model: <strong className="text-gray-800">{behaviorPct}% fake risk</strong>
                  </span>
                )}
                {submitResult.saved && (
                  <span className="ml-auto text-green-700 font-semibold flex items-center gap-1">
                    <CheckCircle size={12}/> Saved to database (ID: {submitResult.review?.id})
                  </span>
                )}
              </div>
            </div>
          )
        })()}


        {/* Submission Error Alert */}
        {submitError && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-3 mb-4 text-sm flex items-center gap-2">
            <AlertTriangle size={16} className="shrink-0" />
            <span>{submitError}</span>
          </div>
        )}

        {/* Authentication Notice or Form */}
        {!user ? (
          <div className="bg-purple-50/70 border border-purple-200/80 rounded-2xl p-6 text-center my-2">
            <div className="inline-flex p-3 bg-purple-100 text-purple-700 rounded-2xl mb-3 shadow-2xs">
              <User size={22} />
            </div>
            <h3 className="text-base font-bold text-gray-900 mb-1">Sign in to Submit a Review</h3>
            <p className="text-sm text-gray-600 max-w-md mx-auto mb-5">
              To guarantee honest community feedback and run predictions through our trained ML fraud model, you must be logged into your We-Review account.
            </p>
            <div className="flex items-center justify-center gap-3">
              <Link
                to="/login"
                className="px-5 py-2.5 bg-purple-600 hover:bg-purple-700 text-white text-sm font-semibold rounded-xl shadow-xs shadow-purple-600/20 transition-all"
              >
                Sign In
              </Link>
              <Link
                to="/register"
                className="px-5 py-2.5 bg-white hover:bg-gray-50 border border-gray-200 text-gray-700 text-sm font-semibold rounded-xl shadow-2xs transition-all"
              >
                Create Account
              </Link>
            </div>
          </div>
        ) : (
          <div>
            {/* Authenticated User Badge */}
            <div className="bg-purple-50/70 border border-purple-100 rounded-xl p-3 mb-5 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full bg-purple-600 text-white font-bold flex items-center justify-center text-xs shadow-2xs">
                  {(user.name || user.username || 'U')[0].toUpperCase()}
                </div>
                <div>
                  <div className="text-xs font-bold text-gray-900 flex items-center gap-1.5">
                    <span>{user.name || user.username}</span>
                    <span className="text-[10px] bg-purple-200/70 text-purple-800 font-semibold px-1.5 py-0.5 rounded">
                      Verified Reviewer
                    </span>
                  </div>
                  <div className="text-[11px] text-gray-500 font-mono">
                    ID: {user.reviewer_id}
                  </div>
                </div>
              </div>
              <span className="text-xs text-purple-700 font-medium bg-white px-2.5 py-1 rounded-md border border-purple-100">
                Authenticated
              </span>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="grid md:grid-cols-2 gap-4">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-1">Your Rating *</label>
                <div className="py-1">
                  <StarRating value={rating} onChange={setRating} />
                </div>
              </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">Review Title *</label>
            <input
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="Short headline summarizing your experience"
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">Review Text *</label>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder="Write detailed feedback about the product..."
              rows={4}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
            />
          </div>

          <div className="md:col-span-2 flex items-center justify-between pt-2">
            <span className="text-xs text-gray-400">
              Target Endpoint: <code className="text-purple-600 font-mono">POST /api/reviews/predict</code>
            </span>
            <button
              type="submit"
              disabled={submitting}
              className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-60 text-white font-semibold px-6 py-2.5 rounded-xl transition-colors shadow-sm"
            >
              {submitting ? (
                <>
                  <RefreshCw size={15} className="animate-spin" />
                  Analyzing with AI...
                </>
              ) : (
                <>
                  <Send size={15} />
                  Submit Review
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    )}
  </div>
</div>
)
}

