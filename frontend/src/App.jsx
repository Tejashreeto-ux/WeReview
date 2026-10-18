
import { useState } from "react";
import "./App.css";

const products = [
  {
    id: 1,
    name: "Wireless Headphones",
    category: "Electronics",
    price: 2499,
    rating: 4.5,
    reviews: 128,
    emoji: "🎧",
    description: "Immersive sound, comfortable fit and long battery life.",
  },
  {
    id: 2,
    name: "Smart Watch",
    category: "Wearables",
    price: 3299,
    rating: 4.3,
    reviews: 96,
    emoji: "⌚",
    description: "Track your activity and stay connected throughout the day.",
  },
  {
    id: 3,
    name: "Everyday Backpack",
    category: "Accessories",
    price: 1199,
    rating: 4.6,
    reviews: 84,
    emoji: "🎒",
    description: "A practical everyday bag with space for all your essentials.",
  },
  {
    id: 4,
    name: "Desk Lamp",
    category: "Home",
    price: 899,
    rating: 4.2,
    reviews: 67,
    emoji: "💡",
    description: "A modern desk lamp for reading, studying and working.",
  },
];

function App() {
  const [search, setSearch] = useState("");
  const [selectedProduct, setSelectedProduct] = useState(products[0]);
  const [review, setReview] = useState("");
  const [result, setResult] = useState("");
  const [cartCount, setCartCount] = useState(0);

  const filteredProducts = products.filter((product) =>
    product.name.toLowerCase().includes(search.toLowerCase())
  );

async function checkReview(event) {
  event.preventDefault();
  setResult("Analyzing review...");
  
  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ review }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

    const score = (data.fake_candidate_score * 100).toFixed(1);

    setResult(
      `${data.message} Fake-candidate score: ${score}%. ${data.notice}`
    );
  } catch (error) {
    setResult(`Error: ${error.message}. Check that the backend is running.`);
  }
}

  

  return (
    <div className="app">
      <header className="navbar">
        <a className="brand" href="#home">WeReview<span>.</span></a>
        <div className="nav-links">
          <a href="#products">Products</a>
          <a href="#reviews">Review checker</a>
          <span className="cart">🛒 Cart ({cartCount})</span>
        </div>
      </header>

      <main>
        <section className="hero" id="home">
          <div className="hero-content">
            <span className="eyebrow">SHOP SMARTER. REVIEW BETTER.</span>
            <h1>Discover products.<br />Trust reviews <span>more.</span></h1>
            <p>
              Explore everyday essentials and use AI-powered review analysis
              to help identify potentially suspicious customer feedback.
            </p>
            <a className="primary-button" href="#products">Explore products →</a>
          </div>
          <div className="hero-art" aria-label="Shopping illustration">
            <div className="hero-circle">🛍️</div>
            <div className="floating-card">✦ Smarter shopping</div>
          </div>
        </section>

        <section className="products-section" id="products">
          <div className="section-heading">
            <div>
              <span className="eyebrow">OUR COLLECTION</span>
              <h2>Featured products</h2>
            </div>
            <input
              className="search"
              placeholder="Search products..."
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              aria-label="Search products"
            />
          </div>

          <div className="product-grid">
            {filteredProducts.map((product) => (
              <article className="product-card" key={product.id}>
                <button
                  className="product-art"
                  onClick={() => {
                    setSelectedProduct(product);
                    setReview("");
                    setResult("");
                  }}
                  aria-label={`Select ${product.name}`}
                >
                  {product.emoji}
                </button>
                <div className="product-info">
                  <span className="category">{product.category}</span>
                  <h3>{product.name}</h3>
                  <p className="description">{product.description}</p>
                  <p className="rating">★ {product.rating} <span>({product.reviews} reviews)</span></p>
                  <div className="product-bottom">
                    <strong>₹{product.price.toLocaleString("en-IN")}</strong>
                    <button
                      className="small-button"
                      onClick={() => setCartCount((count) => count + 1)}
                    >
                      Add to cart
                    </button>
                  </div>
                  <button
                    className="text-button"
                    onClick={() => {
                      setSelectedProduct(product);
                      setReview("");
                      setResult("");
                      document.getElementById("reviews")?.scrollIntoView({
                        behavior: "smooth",
                      });
                    }}
                  >
                    Check a review →
                  </button>
                </div>
              </article>
            ))}
          </div>
          {filteredProducts.length === 0 && (
            <p>No products match your search.</p>
          )}
        </section>

        <section className="review-section" id="reviews">
          <div className="review-intro">
            <span className="eyebrow">WEREVIEW AI</span>
            <h2>Not sure about a review?</h2>
            <p>
              Submit review text for suspicious-review analysis.
              The trained model will be connected to this form next.
            </p>
            <div className="trust-point">✓ AI-assisted review screening</div>
            <div className="trust-point">✓ Risk indicators for closer inspection</div>
          </div>

          <form className="review-form" onSubmit={checkReview}>
            <label htmlFor="product">Selected product</label>
            <select
              id="product"
              value={selectedProduct.id}
              onChange={(event) => {
                setSelectedProduct(
                  products.find((product) => product.id === Number(event.target.value))
                );
                setReview("");
                setResult("");
              }}
            >
              {products.map((product) => (
                <option value={product.id} key={product.id}>{product.name}</option>
              ))}
            </select>

            <label htmlFor="review">Customer review</label>
            <textarea
              id="review"
              rows="5"
              minLength="10"
              required
              placeholder="Paste the customer review you want to check..."
              value={review}
              onChange={(event) => setReview(event.target.value)}
            />

            <button className="primary-button submit-button" type="submit">
              Analyze review →
            </button>

            {result && <p className="result-message" role="status">{result}</p>}
          </form>
        </section>
      </main>

      <footer>
        <a className="brand" href="#home">WeReview<span>.</span></a>
        <p>AI-assisted review screening for more informed shopping.</p>
        <small>Demo storefront · Product listings and prices are illustrative.</small>
      </footer>
    </div>
  );
}

export default App;

