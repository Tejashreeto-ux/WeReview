const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { pool, initDatabase } = require('./db');

const cookieParser = require('cookie-parser');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

const FLASK_API_URL = 'http://127.0.0.1:5001';

const JWT_SECRET = process.env.JWT_SECRET || 'we_review_super_secret_jwt_key_2025';
const COOKIE_NAME = 'token';

const allowedOrigins = [
  'http://localhost:3000',
  'http://127.0.0.1:3000',
  'http://localhost:5173',
  'http://127.0.0.1:5173',
];

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);
      if (
        allowedOrigins.includes(origin) ||
        origin.startsWith('http://localhost:') ||
        origin.startsWith('http://127.0.0.1:')
      ) {
        return callback(null, origin);
      }
      return callback(null, origin);
    },
    credentials: true,
  })
);
app.use(cookieParser());
app.use(express.json());

// Helper for HTTP-only cookie options
function getCookieOptions() {
  return {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    path: '/',
  };
}

// Authentication middleware to extract and verify user from JWT cookie or Authorization header
async function authMiddleware(req, res, next) {
  let token = req.cookies ? req.cookies[COOKIE_NAME] : null;

  if (!token && req.headers.authorization && req.headers.authorization.startsWith('Bearer ')) {
    token = req.headers.authorization.split(' ')[1];
  }

  if (!token) {
    req.user = null;
    return next();
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    const [rows] = await pool.query(
      'SELECT id, reviewer_id, username, name, email, created_at FROM users WHERE id = ?',
      [decoded.id]
    );

    if (rows && rows.length > 0) {
      req.user = rows[0];
    } else {
      req.user = null;
    }
  } catch (err) {
    req.user = null;
  }

  next();
}

app.use(authMiddleware);

// List of required behavior features expected by the trained model
const REQUIRED_BEHAVIOR_FEATURES = [
  'review_rating_numeric',
  'review_year',
  'review_month',
  'review_day_of_week',
  'review_day_of_year',
  'reviewer_prior_review_count',
  'reviewer_prior_mean_rating',
  'reviewer_prior_unique_related',
  'reviewer_days_since_prior_review',
  'product_prior_review_count',
  'product_prior_mean_rating',
  'product_prior_unique_related',
  'product_days_since_prior_review',
];

/**
 * Basic Health Check for Express backend + Database connection status
 */
app.get('/health', async (req, res) => {
  let dbStatus = 'disconnected';
  let tablesList = [];

  try {
    const [rows] = await pool.query('SHOW TABLES;');
    dbStatus = 'connected';
    tablesList = rows.map((r) => Object.values(r)[0]);
  } catch (err) {
    dbStatus = `error: ${err.message}`;
  }

  res.status(200).json({
    status: 'healthy',
    service: 'We-Review Express Backend',
    database: {
      status: dbStatus,
      tables: tablesList,
    },
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

/**
 * Endpoint to explicitly test and inspect MySQL tables
 */
app.get('/db-check', async (req, res) => {
  try {
    const [tables] = await pool.query('SHOW TABLES;');
    const tableNames = tables.map((t) => Object.values(t)[0]);

    const schemaDetails = {};
    for (const name of tableNames) {
      const [columns] = await pool.query(`DESCRIBE \`${name}\`;`);
      schemaDetails[name] = columns.map((col) => ({
        field: col.Field,
        type: col.Type,
        null: col.Null,
        key: col.Key,
      }));
    }

    return res.status(200).json({
      status: 'connected',
      database: 'we_review',
      tables: tableNames,
      schemas: schemaDetails,
    });
  } catch (error) {
    return res.status(500).json({
      status: 'error',
      message: 'Failed to query database tables',
      error: error.message,
    });
  }
});

/**
 * Test endpoint calling the Flask AI API health check
 */
app.get('/flask-health', async (req, res) => {
  const targetUrl = `${FLASK_API_URL}/health`;

  try {
    const response = await fetch(targetUrl, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      return res.status(502).json({
        status: 'error',
        message: `Flask API responded with status ${response.status}`,
        endpoint: targetUrl,
      });
    }

    const data = await response.json();
    return res.status(200).json({
      status: 'connected',
      flaskEndpoint: targetUrl,
      flaskData: data,
    });
  } catch (error) {
    let errorMessage = 'Failed to reach Flask API';
    if (error.name === 'TimeoutError') {
      errorMessage = 'Flask API connection timed out';
    } else if (error.code === 'ECONNREFUSED') {
      errorMessage = 'Flask API connection refused (service may not be running)';
    }

    return res.status(503).json({
      status: 'unavailable',
      error: errorMessage,
      details: error.message,
      targetUrl,
    });
  }
});

/**
 * POST /api/reviews/predict
 * Receives review title, text, rating, and behavior features,
 * validates the input, and forwards to Flask AI API.
 */
app.post('/api/reviews/predict', async (req, res) => {
  const body = req.body;

  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return res.status(400).json({
      status: 'error',
      message: 'Invalid request body: Expected a JSON object.',
    });
  }

  // 1. Validate Review Text
  const reviewText = body.review_text;
  if (!reviewText || typeof reviewText !== 'string' || !reviewText.trim()) {
    return res.status(400).json({
      status: 'error',
      message: "Validation error: 'review_text' is required and must be a non-empty string.",
    });
  }

  // 2. Validate Review Title (optional but must be string if given)
  const reviewTitle = body.review_title ?? '';
  if (typeof reviewTitle !== 'string') {
    return res.status(400).json({
      status: 'error',
      message: "Validation error: 'review_title' must be a string.",
    });
  }

  // 3. Handle review rating numeric (supports 'review_rating_numeric' or shorthand 'review_rating' or 'rating')
  let ratingNumeric = body.review_rating_numeric ?? body.review_rating ?? body.rating;
  if (ratingNumeric !== undefined && ratingNumeric !== null && ratingNumeric !== '') {
    const parsedRating = Number(ratingNumeric);
    if (isNaN(parsedRating) || parsedRating < 1 || parsedRating > 5) {
      return res.status(400).json({
        status: 'error',
        message: "Validation error: rating must be a valid number between 1.0 and 5.0.",
      });
    }
    ratingNumeric = parsedRating;
  } else {
    ratingNumeric = null;
  }

  // 4. Determine reviewer ID and review date features
  // Enforce authentication: reviewerId MUST come from authenticated user session
  if (!req.user) {
    return res.status(401).json({
      status: 'error',
      message: 'Authentication required. Please log in to submit a review.',
    });
  }

  // Derive reviewerId strictly from authenticated user - NEVER trust frontend input
  const reviewerId = req.user.reviewer_id;
  const asin = body.asin || 'unknown_product';
  const reviewDate = body.review_date ? new Date(body.review_date) : new Date();

  const year = reviewDate.getFullYear();
  const month = reviewDate.getMonth() + 1;
  const dayOfWeek = (reviewDate.getDay() + 6) % 7; // Monday=0, Sunday=6 (matching pandas dt.day_of_week)
  const startOfYear = new Date(Date.UTC(year, 0, 1));
  const reviewUtc = new Date(Date.UTC(year, reviewDate.getMonth(), reviewDate.getDate()));
  const dayOfYear = Math.floor((reviewUtc - startOfYear) / (1000 * 60 * 60 * 24)) + 1;

  // 5. Calculate reviewer history strictly matching train_fake_review_models.py
  // In training: if entity has 0 prior reviews, prior_mean_rating and days_since_prior are NaN (null),
  // which SimpleImputer fills with median. It must NOT default to the current rating.
  let reviewerPriorCount = body.reviewer_prior_review_count;
  let reviewerPriorMean = body.reviewer_prior_mean_rating;
  let reviewerPriorUnique = body.reviewer_prior_unique_related;
  let reviewerDaysSince = body.reviewer_days_since_prior_review;

  if (reviewerPriorCount === undefined || reviewerPriorCount === null) {
    try {
      const [rows] = await pool.query(
        `SELECT 
           COUNT(*) AS count, 
           AVG(review_rating) AS mean_rating, 
           COUNT(DISTINCT asin) AS unique_related, 
           MAX(review_date) AS last_date 
         FROM reviews 
         WHERE reviewer_id = ? AND review_date < ?`,
        [reviewerId, reviewDate]
      );
      if (rows && rows[0] && rows[0].count > 0) {
        reviewerPriorCount = Number(rows[0].count);
        reviewerPriorMean = rows[0].mean_rating !== null ? Number(rows[0].mean_rating) : null;
        reviewerPriorUnique = Number(rows[0].unique_related);
        if (rows[0].last_date) {
          const diffMs = reviewDate - new Date(rows[0].last_date);
          reviewerDaysSince = Math.max(0, diffMs / (1000 * 60 * 60 * 24));
        } else {
          reviewerDaysSince = null;
        }
      } else {
        reviewerPriorCount = 0;
        reviewerPriorMean = null; // null -> np.nan -> SimpleImputer median
        reviewerPriorUnique = 0;
        reviewerDaysSince = null;
      }
    } catch (e) {
      reviewerPriorCount = 0;
      reviewerPriorMean = null;
      reviewerPriorUnique = 0;
      reviewerDaysSince = null;
    }
  } else {
    reviewerPriorCount = Number(reviewerPriorCount);
    reviewerPriorMean = reviewerPriorMean !== undefined && reviewerPriorMean !== null && reviewerPriorMean !== '' ? Number(reviewerPriorMean) : null;
    reviewerPriorUnique = Number(reviewerPriorUnique ?? 0);
    reviewerDaysSince = reviewerDaysSince !== undefined && reviewerDaysSince !== null && reviewerDaysSince !== '' ? Number(reviewerDaysSince) : null;
  }

  // 6. Calculate product history strictly matching train_fake_review_models.py
  let productPriorCount = body.product_prior_review_count;
  let productPriorMean = body.product_prior_mean_rating;
  let productPriorUnique = body.product_prior_unique_related;
  let productDaysSince = body.product_days_since_prior_review;

  if (productPriorCount === undefined || productPriorCount === null) {
    try {
      const [rows] = await pool.query(
        `SELECT 
           COUNT(*) AS count, 
           AVG(review_rating) AS mean_rating, 
           COUNT(DISTINCT reviewer_id) AS unique_related, 
           MAX(review_date) AS last_date 
         FROM reviews 
         WHERE asin = ? AND review_date < ?`,
        [asin, reviewDate]
      );
      if (rows && rows[0] && rows[0].count > 0) {
        productPriorCount = Number(rows[0].count);
        productPriorMean = rows[0].mean_rating !== null ? Number(rows[0].mean_rating) : null;
        productPriorUnique = Number(rows[0].unique_related);
        if (rows[0].last_date) {
          const diffMs = reviewDate - new Date(rows[0].last_date);
          productDaysSince = Math.max(0, diffMs / (1000 * 60 * 60 * 24));
        } else {
          productDaysSince = null;
        }
      } else {
        productPriorCount = 0;
        productPriorMean = null; // null -> np.nan -> SimpleImputer median (3.75)
        productPriorUnique = 0;
        productDaysSince = null;
      }
    } catch (e) {
      productPriorCount = 0;
      productPriorMean = null;
      productPriorUnique = 0;
      productDaysSince = null;
    }
  } else {
    productPriorCount = Number(productPriorCount);
    productPriorMean = productPriorMean !== undefined && productPriorMean !== null && productPriorMean !== '' ? Number(productPriorMean) : null;
    productPriorUnique = Number(productPriorUnique ?? 0);
    productDaysSince = productDaysSince !== undefined && productDaysSince !== null && productDaysSince !== '' ? Number(productDaysSince) : null;
  }

  const flaskPayload = {
    review_title: reviewTitle,
    review_text: reviewText,
    review_rating_numeric: ratingNumeric,
    review_year: body.review_year ?? year,
    review_month: body.review_month ?? month,
    review_day_of_week: body.review_day_of_week ?? dayOfWeek,
    review_day_of_year: body.review_day_of_year ?? dayOfYear,
    reviewer_prior_review_count: reviewerPriorCount,
    reviewer_prior_mean_rating: reviewerPriorMean,
    reviewer_prior_unique_related: reviewerPriorUnique,
    reviewer_days_since_prior_review: reviewerDaysSince,
    product_prior_review_count: productPriorCount,
    product_prior_mean_rating: productPriorMean,
    product_prior_unique_related: productPriorUnique,
    product_days_since_prior_review: productDaysSince,
  };

  // Verify all numeric features are valid numbers or null
  const invalidFields = [];
  for (const field of REQUIRED_BEHAVIOR_FEATURES) {
    const val = flaskPayload[field];
    if (val !== null && val !== undefined && val !== '') {
      if (isNaN(Number(val))) {
        invalidFields.push(field);
      } else {
        flaskPayload[field] = Number(val);
      }
    } else {
      flaskPayload[field] = null;
    }
  }

  if (invalidFields.length > 0) {
    return res.status(400).json({
      status: 'error',
      message: `Invalid numeric values provided for behavior fields: ${invalidFields.join(', ')}`,
    });
  }

  // 5. Log exact request and forward to Flask AI API
  const flaskPredictUrl = `${FLASK_API_URL}/predict`;

  console.log('\n======================================================');
  console.log('[1. Frontend -> Node.js] Exact request received at POST /api/reviews/predict:');
  console.log(JSON.stringify(body, null, 2));
  console.log('\n[2. Node.js -> Flask] Exact payload forwarded to ' + flaskPredictUrl + ':');
  console.log(JSON.stringify(flaskPayload, null, 2));
  console.log('======================================================\n');

  try {
    const response = await fetch(flaskPredictUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(flaskPayload),
      signal: AbortSignal.timeout(10000), // 10s timeout
    });

    const data = await response.json();

    console.log('\n[3. Flask -> Node.js] Exact response received:');
    console.log(JSON.stringify(data, null, 2));

    if (!response.ok) {
      return res.status(response.status).json({
        status: 'error',
        message: data.error || 'Flask AI API returned an error.',
        details: data,
      });
    }

    // Check if the user has already submitted a review for this exact same product
    let duplicateCount = 0;
    try {
      const [dupeRows] = await pool.query(
        'SELECT COUNT(*) AS cnt FROM reviews WHERE reviewer_id = ? AND asin = ?',
        [reviewerId, asin]
      );
      duplicateCount = Number(dupeRows[0]?.cnt || 0);
    } catch (e) {
      duplicateCount = 0;
    }

    // If the same user has already reviewed this product, it is review duplication/spam!
    if (duplicateCount > 0) {
      console.log(`[SPAM DETECTION] Reviewer ${reviewerId} has already submitted ${duplicateCount} reviews for product ${asin}. Tagging as FAKE spam.`);
      data.prediction = 'fake';
      data.fake_probability = Math.max(0.85, Number(data.fake_probability));
      data.genuine_probability = Number((1.0 - data.fake_probability).toFixed(4));
      if (!data.details) data.details = {};
      data.details.duplicate_review_spam = true;
      data.details.prior_reviews_for_product = duplicateCount;
    }

    // 7. Save to MySQL database (reviews table)
    let savedRecord = null;
    try {
      const [insertResult] = await pool.query(
        `INSERT INTO reviews 
          (reviewer_id, asin, review_title, review_text, review_rating, review_date, fake_probability, prediction)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          reviewerId,
          asin,
          reviewTitle,
          reviewText,
          ratingNumeric,
          reviewDate,
          data.fake_probability,
          data.prediction,
        ]
      );

      savedRecord = {
        id: insertResult.insertId,
        reviewer_id: reviewerId,
        asin: asin,
        review_title: reviewTitle,
        review_text: reviewText,
        review_rating: ratingNumeric,
        fake_probability: data.fake_probability,
        prediction: data.prediction,
      };

      console.log('\n[4. Node.js -> MySQL] Saved review record:');
      console.log(JSON.stringify(savedRecord, null, 2));
      console.log('======================================================\n');
    } catch (dbError) {
      console.error('[DB Error] Failed to persist review to MySQL:', dbError.message);
      return res.status(500).json({
        status: 'error',
        message: 'Prediction generated successfully, but failed to save review into MySQL database.',
        prediction: data.prediction,
        fake_probability: data.fake_probability,
        database_error: dbError.message,
      });
    }

    return res.status(200).json({
      status: 'success',
      prediction: data.prediction,
      fake_probability: data.fake_probability,
      genuine_probability: data.genuine_probability,
      model_used: data.model_used,
      saved: true,
      review: savedRecord,
      details: data.details,
    });
  } catch (error) {
    let errorMessage = 'Failed to connect to Flask AI API.';
    if (error.name === 'TimeoutError') {
      errorMessage = 'Flask AI API prediction request timed out.';
    } else if (error.code === 'ECONNREFUSED') {
      errorMessage = 'Flask AI API connection refused (service at 127.0.0.1:5000 may not be running).';
    }

    return res.status(503).json({
      status: 'unavailable',
      error: errorMessage,
      details: error.message,
      targetUrl: flaskPredictUrl,
    });
  }
});

/**
 * GET /api/reviews
 * Fetches reviews from MySQL database, optionally filtered by product ASIN.
 * Query parameters:
 *  - asin: (optional) Filter reviews by product ASIN
 *
 * GET /api/reviews/:asin
 * Convenient route to fetch reviews specifically for a given ASIN.
 */
async function handleGetReviews(req, res) {
  const asin = req.params.asin || req.query.asin;

  try {
    let query = `
      SELECT 
        id, 
        reviewer_id, 
        asin, 
        review_title, 
        review_text, 
        CAST(review_rating AS DECIMAL(3,2)) AS review_rating, 
        review_date, 
        CAST(fake_probability AS DECIMAL(5,4)) AS fake_probability, 
        prediction, 
        created_at
      FROM reviews
    `;
    const params = [];

    if (asin && String(asin).trim()) {
      query += ' WHERE asin = ?';
      params.push(String(asin).trim());
    }

    query += ' ORDER BY created_at DESC, id DESC';

    const [rows] = await pool.query(query, params);

    const formattedReviews = rows.map((r) => ({
      id: r.id,
      reviewer_id: r.reviewer_id,
      asin: r.asin,
      review_title: r.review_title || '',
      review_text: r.review_text || '',
      review_rating: r.review_rating !== null ? Number(r.review_rating) : null,
      review_date: r.review_date,
      fake_probability: r.fake_probability !== null ? Number(r.fake_probability) : null,
      prediction: r.prediction,
      created_at: r.created_at,
    }));

    return res.status(200).json({
      status: 'success',
      count: formattedReviews.length,
      asin: asin ? String(asin).trim() : null,
      reviews: formattedReviews,
    });
  } catch (error) {
    console.error('[DB Error] Failed to fetch reviews from MySQL:', error.message);
    return res.status(500).json({
      status: 'error',
      message: 'Failed to fetch reviews from MySQL database.',
      error: error.message,
    });
  }
}

app.get('/api/reviews', handleGetReviews);
app.get('/api/reviews/:asin', handleGetReviews);

/**
 * ========================================================
 * AUTHENTICATION ENDPOINTS
 * ========================================================
 */

/**
 * POST /api/auth/register
 * Registration fields: name, email, password, and confirm password.
 */
app.post('/api/auth/register', async (req, res) => {
  try {
    const { name, email, password, confirm_password, confirmPassword } = req.body || {};
    const confirm = confirm_password !== undefined ? confirm_password : confirmPassword;

    // Validate name
    if (!name || typeof name !== 'string' || !name.trim()) {
      return res.status(400).json({
        status: 'error',
        message: 'Name is required.',
      });
    }

    // Validate email
    if (!email || typeof email !== 'string' || !email.trim()) {
      return res.status(400).json({
        status: 'error',
        message: 'Email is required.',
      });
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const cleanEmail = email.trim().toLowerCase();
    if (!emailRegex.test(cleanEmail)) {
      return res.status(400).json({
        status: 'error',
        message: 'Please provide a valid email address.',
      });
    }

    // Validate password
    if (!password || typeof password !== 'string' || password.length < 6) {
      return res.status(400).json({
        status: 'error',
        message: 'Password must be at least 6 characters long.',
      });
    }

    // Validate confirm password
    if (confirm === undefined || confirm === null || password !== confirm) {
      return res.status(400).json({
        status: 'error',
        message: 'Passwords do not match.',
      });
    }

    // Check if user with this email already exists
    const [existing] = await pool.query('SELECT id FROM users WHERE email = ?', [cleanEmail]);
    if (existing && existing.length > 0) {
      return res.status(400).json({
        status: 'error',
        message: 'An account with this email already exists.',
      });
    }

    // Hash password using bcrypt
    const passwordHash = await bcrypt.hash(password, 10);

    // Generate unique reviewer_id and username
    const randomSuffix = Math.random().toString(36).substring(2, 8).toUpperCase();
    const reviewerId = `REV_${Date.now().toString(36).toUpperCase()}_${randomSuffix}`;
    const cleanName = name.trim();
    const username = cleanName.toLowerCase().replace(/[^a-z0-9]/g, '_').substring(0, 50) || 'user';

    const [insertResult] = await pool.query(
      `INSERT INTO users (reviewer_id, username, name, email, password_hash)
       VALUES (?, ?, ?, ?, ?)`,
      [reviewerId, username, cleanName, cleanEmail, passwordHash]
    );

    const newUser = {
      id: insertResult.insertId,
      reviewer_id: reviewerId,
      username: username,
      name: cleanName,
      email: cleanEmail,
    };

    // Generate JWT token
    const token = jwt.sign(
      { id: newUser.id, reviewer_id: newUser.reviewer_id, email: newUser.email },
      JWT_SECRET,
      { expiresIn: '7d' }
    );

    // Set HTTP-only cookie
    res.cookie(COOKIE_NAME, token, getCookieOptions());

    return res.status(201).json({
      status: 'success',
      message: 'Registration successful',
      user: newUser,
    });
  } catch (error) {
    console.error('[Auth Error] Registration failed:', error);
    return res.status(500).json({
      status: 'error',
      message: 'Registration failed due to a server error.',
      error: error.message,
    });
  }
});

/**
 * POST /api/auth/login
 * Login fields: email and password.
 */
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body || {};

    if (!email || !password) {
      return res.status(400).json({
        status: 'error',
        message: 'Email and password are required.',
      });
    }

    const cleanEmail = String(email).trim().toLowerCase();

    const [rows] = await pool.query(
      'SELECT id, reviewer_id, username, name, email, password_hash, created_at FROM users WHERE email = ?',
      [cleanEmail]
    );

    if (!rows || rows.length === 0) {
      return res.status(401).json({
        status: 'error',
        message: 'Invalid email or password.',
      });
    }

    const user = rows[0];

    if (!user.password_hash) {
      return res.status(401).json({
        status: 'error',
        message: 'This account does not have a password configured.',
      });
    }

    const isMatch = await bcrypt.compare(String(password), user.password_hash);
    if (!isMatch) {
      return res.status(401).json({
        status: 'error',
        message: 'Invalid email or password.',
      });
    }

    const token = jwt.sign(
      { id: user.id, reviewer_id: user.reviewer_id, email: user.email },
      JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.cookie(COOKIE_NAME, token, getCookieOptions());

    return res.status(200).json({
      status: 'success',
      message: 'Login successful',
      user: {
        id: user.id,
        reviewer_id: user.reviewer_id,
        username: user.username,
        name: user.name || user.username,
        email: user.email,
        created_at: user.created_at,
      },
    });
  } catch (error) {
    console.error('[Auth Error] Login failed:', error);
    return res.status(500).json({
      status: 'error',
      message: 'Login failed due to a server error.',
      error: error.message,
    });
  }
});

/**
 * GET /api/auth/me
 * Retrieves current authenticated user session from HTTP-only cookie.
 */
app.get('/api/auth/me', (req, res) => {
  if (!req.user) {
    return res.status(401).json({
      status: 'error',
      authenticated: false,
      message: 'Not authenticated',
    });
  }

  return res.status(200).json({
    status: 'success',
    authenticated: true,
    user: {
      id: req.user.id,
      reviewer_id: req.user.reviewer_id,
      username: req.user.username,
      name: req.user.name || req.user.username,
      email: req.user.email,
      created_at: req.user.created_at,
    },
  });
});

/**
 * POST /api/auth/logout
 * Clears HTTP-only token cookie.
 */
app.post('/api/auth/logout', (req, res) => {
  res.clearCookie(COOKIE_NAME, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
  });

  return res.status(200).json({
    status: 'success',
    message: 'Logged out successfully',
  });
});


// Initialize database tables and start Express Server
async function startServer() {
  try {
    await initDatabase();
  } catch (err) {
    console.error('[DB Warning] Initial DB migration check failed:', err.message);
  }

  app.listen(PORT, () => {
    console.log(`\n=== We-Review Express Backend ===`);
    console.log(`Server running at: http://localhost:${PORT}`);
    console.log(`Health check:     http://localhost:${PORT}/health`);
    console.log(`Database check:   http://localhost:${PORT}/db-check`);
    console.log(`Flask test check: http://localhost:${PORT}/flask-health`);
    console.log(`Predict endpoint: POST http://localhost:${PORT}/api/reviews/predict`);
    console.log(`Reviews endpoint: GET  http://localhost:${PORT}/api/reviews`);
    console.log(`Connected Flask:  ${FLASK_API_URL}\n`);
  });
}

startServer();
