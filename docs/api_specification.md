# REST API Specification & Endpoint Contracts

## Base URL & Versioning
- **Base URL**: `http://localhost:8000/api/v1`
- **Content Type**: `application/json`
- **Response Format Wrapper**:
  ```json
  {
    "success": true,
    "data": { ... },
    "message": "Operation successful",
    "timestamp": "2026-07-24T18:00:00Z"
  }
  ```

---

## Endpoint Summary

### System & Health Endpoints

#### `GET /api/v1/health`
- **Summary**: Service Health Status Check
- **Auth**: None
- **Response 200 OK**:
  ```json
  {
    "success": true,
    "data": {
      "status": "healthy",
      "version": "0.1.0",
      "environment": "development",
      "timestamp": "2026-07-24T18:00:00Z"
    },
    "message": "Service is healthy"
  }
  ```

#### `GET /api/v1/health/ready`
- **Summary**: Readiness Probe
- **Auth**: None
- **Response 200 OK**:
  ```json
  {
    "success": true,
    "data": {
      "ready": true,
      "checks": {
        "database": "ok"
      }
    }
  }
  ```

---

### Authentication Feature (`/api/v1/auth`)

#### `POST /api/v1/auth/register`
- **Summary**: Register a new user account
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "SecurePassword123!",
    "full_name": "John Doe"
  }
  ```
- **Response 201 Created**:
  ```json
  {
    "success": true,
    "data": {
      "id": "u49d2a-...",
      "email": "user@example.com",
      "full_name": "John Doe",
      "is_active": true,
      "is_superuser": false,
      "created_at": "2026-07-24T18:00:00Z"
    },
    "message": "User registered successfully"
  }
  ```

#### `POST /api/v1/auth/login`
- **Summary**: Authenticate user and receive JWT pair
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "SecurePassword123!"
  }
  ```
- **Response 200 OK**:
  ```json
  {
    "success": true,
    "data": {
      "access_token": "eyJhbGciOi...",
      "refresh_token": "eyJhbGciOi...",
      "token_type": "bearer",
      "expires_in": 1800
    },
    "message": "Login successful"
  }
  ```

#### `POST /api/v1/auth/refresh`
- **Summary**: Obtain fresh access token using valid refresh token
- **Request Body**:
  ```json
  {
    "refresh_token": "eyJhbGciOi..."
  }
  ```
- **Response 200 OK**: Returns updated access token pair.

#### `GET /api/v1/auth/me`
- **Summary**: Retrieve current user profile
- **Headers**: `Authorization: Bearer <access_token>`
- **Response 200 OK**: Returns current `UserResponse` object.

---

### Database Connections Feature (`/api/v1/connections`)

#### `GET /api/v1/connections`
- **Summary**: List user's target database connections
- **Auth**: Bearer Token

#### `POST /api/v1/connections`
- **Summary**: Create new database connection target
- **Request Body**:
  ```json
  {
    "name": "Production Postgres",
    "dialect": "postgresql",
    "host": "db.example.com",
    "port": 5432,
    "database_name": "sales_db",
    "username": "readonly_user",
    "password": "SecretPassword"
  }
  ```

#### `POST /api/v1/connections/{id}/test`
- **Summary**: Test database connection connectivity

---

### Schema Introspection Feature (`/api/v1/schema`)

#### `POST /api/v1/schema/{connection_id}/sync`
- **Summary**: Introspect and update stored schema metadata for a database connection

#### `GET /api/v1/schema/{connection_id}`
- **Summary**: Get structured schema tables, columns, and relationship metadata

---

### Text-to-SQL Query Feature (`/api/v1/query`)

#### `POST /api/v1/query/generate`
- **Summary**: Convert natural language prompt to validated SQL query using Gemini LLM
- **Auth**: Bearer Token
- **Request Body**:
  ```json
  {
    "connection_id": "conn-1234",
    "prompt": "Show top 5 customers by total order amount in 2025"
  }
  ```
- **Response 200 OK**:
  ```json
  {
    "success": true,
    "data": {
      "query_id": "q-9876",
      "natural_language_prompt": "Show top 5 customers by total order amount in 2025",
      "generated_sql": "SELECT c.customer_name, SUM(o.total_amount) AS total_spent FROM customers c JOIN orders o ON c.id = o.customer_id WHERE strftime('%Y', o.order_date) = '2025' GROUP BY c.id ORDER BY total_spent DESC LIMIT 5;",
      "is_valid": true,
      "explanation": "Joined customers and orders tables, filtered by year 2025, aggregated order totals per customer, and limited output to top 5."
    }
  }
  ```

#### `POST /api/v1/query/execute`
- **Summary**: Safely execute validated read-only SQL query against target database
- **Request Body**:
  ```json
  {
    "connection_id": "conn-1234",
    "sql": "SELECT c.customer_name, SUM(o.total_amount) AS total_spent FROM customers c JOIN orders o ON c.id = o.customer_id WHERE strftime('%Y', o.order_date) = '2025' GROUP BY c.id ORDER BY total_spent DESC LIMIT 5;"
  }
  ```
- **Response 200 OK**:
  ```json
  {
    "success": true,
    "data": {
      "columns": ["customer_name", "total_spent"],
      "rows": [
        ["Acme Corp", 154200.50],
        ["Globex Inc", 98400.00]
      ],
      "row_count": 2,
      "execution_time_ms": 42
    }
  }
  ```
