# 安全验证方式
Web authentication and Authorization

1. Authentication: Who are you?
2. Authorization: What can you do?

> Reference: https://testdriven.io/blog/web-authentication-methods/

几种基本的安全验证方式：
1. [HTTP Basic Authentication](#HTTP-Basic-Authentication)
2. [HTTP Digest Authentication](#HTTP-Digest-Authentication)
3. [Session-based Auth](#Session-based-Auth)
4. [Token-Based Authentication](#Token-Based-Authentication)
5. [One Time Passwords](#One-Time-Passwords)
6. [OAuth and OpenID](#OAuth-and-OpenID)

## HTTP Basic Authentication

用户登陆信息会以明文的方式放在HTTP 请求头`Authorization: Basic `，例如： `username:password`。

Pros:
- Easy to implement

Cons:
- Expose sensitive data even if it's Base64 data.
- Credential must sent in each request

## HTTP Digest Authentication
HTTP Digest Authentication (or Digest Access Authentication) is a more secure form of HTTP Basic Auth. The main difference is that the password is sent in MD5 hashed form rather than in plain text, so it's more secure than Basic Auth.

Pros
- More secure than Basic auth since the password is not sent in plain text.
- Easy to implement.
- Supported by all major browsers.

Cons
- Credentials must be sent with every request.
- User can only be logged out by rewriting the credentials with an invalid one.
- Compared to Basic auth, passwords are less secure on the server since bcrypt can't be used.
- Vulnerable to [man-in-the-middle attacks](https://www.veracode.com/security/man-middle-attack).

## Session-based Auth

With session-based auth (or session cookie auth or cookie-based auth), the user's state is stored on the server. It does not require the user to provide a username or a password with each request. Instead, after logging in, the server validates the credentials. If valid, it generates a session, stores it in a session store, and then sends the session ID back to the browser. The browser stores the session ID as a cookie, which gets sent anytime a request is made to the server.

Session-based auth is **stateful**. Each time a client requests the server, the server must locate the session in memory in order to tie the session ID back to the associated user.

Pros
- Faster subsequent logins, as the credentials are not required.
- Improved user experience.
- Fairly easy to implement. Many frameworks (like Django) provide this feature out-of-the-box.

Cons
- It's stateful. The server keeps track of each session on the server-side. The session store, used for storing user session information, needs to be shared across multiple services to enable authentication. Because of this, it doesn't work well for RESTful services, since REST is a stateless protocol.
- Cookies are sent with every request, even if it does not require authentication
- Vulnerable to CSRF attacks.

## Token-Based Authentication
This method uses tokens to authenticate users instead of cookies. The user authenticates using valid credentials and the server returns a signed token. This token can be used for subsequent requests.

The most commonly used token is a [JSON Web Token (JWT)](https://jwt.io/). A JWT consists of three parts:

- **Header** (includes the token type and the hashing algorithm used)
- **Payload** (includes the claims, which are statements about the subject)
- **Signature** (used to verify that the message wasn't changed along the way)

All three are base64 encoded and concatenated using a . and hashed. Since they are encoded, anyone can decode and read the message. But only authentic users can produce valid signed tokens. The token is authenticated using the Signature, which is signed with a **private key**.

Pros
- It's stateless. The server doesn't need to store the token as it can be validated using the signature. This makes the request faster as a database lookup is not required.
- Suited for a microservices architecture, where multiple services require authentication. All we need to configure at each end is how to handle the token and the token secret.

Cons
- Depending on how the token is saved on the client, it can lead to XSS (via localStorage) or CSRF (via cookies) attacks.
- Tokens cannot be deleted. They can only expire. This means that if the token gets leaked, an attacker can misuse it until expiry. Thus, it's important to set token expiry to something very small, like 15 minutes.
- Refresh tokens need to be set up to automatically issue tokens at expiry.
- One way to delete tokens is to create a database for blacklisting tokens. This adds extra overhead to the microservice architecture and introduces state.
## One Time Passwords
One time passwords (OTPs) are commonly used as confirmation for authentication. OTPs are randomly generated codes that can be used to verify if the user is who they claim to be. Its often used after user credentials are verified for apps that leverage two-factor authentication.

**To use OTP, a trusted system must be present. This trusted system could be a verified email or mobile number.**

Modern OTPs are stateless. They can be verified using multiple methods. While there are a few different types of OTPs, Time-based OTPs (TOTPs) is arguably the most common type. Once generated, they expire after a period of time.

Since you get an added layer of security, OTPs are recommended for apps that involve highly sensitive data, like online banking and other financial services.

Pros
- Adds an extra layer of protection.
- No danger that a stolen password can be used for multiple sites or services that also implement OTPs.

Cons
- You need to store the seed used for generating OTPs.
- OTP agents like Google Authenticator are difficult to set up again if you lose the recovery code.
- Problems arise when the trusted device is not available (dead battery, network error, etc.). Because of this, a backup device is typically required which adds an additional attack vector.
## OAuth and OpenID

Pros
- Improved security.
- Easier and faster log in flows since there's no need to create and remember a username or password.
- In case of a security breach, no third-party damage will occur, as the authentication is passwordless.

Cons
- Your application now depends on another app, outside of your control. If the OpenID system is down, users won't be able to log in.
- People often tend to ignore the permissions requested by OAuth applications.
- Users that don't have accounts on the OpenID providers that you have configured won't be able to access your application. The best approach is to implement both -- e.g., username and password and OpenID -- and let the user choose.