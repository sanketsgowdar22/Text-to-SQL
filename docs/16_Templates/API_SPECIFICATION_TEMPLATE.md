# API Specification Template

> **Endpoint**: `METHOD /api/v1/path`
> **Feature**: [Feature Name]
> **Auth Required**: Yes / No
> **Phase**: Phase XX

---

## Summary

<!-- Brief description of what this endpoint does -->

---

## Request

### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes / No | `Bearer <access_token>` |
| `Content-Type` | Yes | `application/json` |

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| | | | |

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| | | | |

### Request Body

```json
{
    "field": "value"
}
```

| Field | Type | Required | Validation | Description |
|-------|------|----------|-----------|-------------|
| | | | | |

---

## Response

### Success Response (200 OK)

```json
{
    "success": true,
    "data": {},
    "message": "Operation successful"
}
```

### Error Responses

| Status | Error Type | Condition |
|--------|-----------|-----------|
| 400 | BadRequest | Invalid input |
| 401 | Unauthorized | Missing/invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | NotFound | Resource not found |
| 409 | Conflict | Resource already exists |
| 422 | ValidationError | Schema validation failed |

---

## Examples

### cURL

```bash
curl -X METHOD http://localhost:8000/api/v1/path \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"field": "value"}'
```

---

## Notes

<!-- Any additional implementation notes, edge cases, or limitations -->
