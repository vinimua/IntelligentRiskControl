import { NextRequest, NextResponse } from "next/server";

const DEFAULT_MODEL_OPS_API_BASE =
  process.env.NEXT_PUBLIC_MODEL_OPS_API_BASE ?? "http://localhost:8001";

type RouteContext = {
  params: {
    path?: string[];
  };
};

function resolveBackendUrl(request: NextRequest, path: string[]): string {
  const configuredBase =
    request.headers.get("x-modelops-api-base") || DEFAULT_MODEL_OPS_API_BASE;
  const base = configuredBase.trim().replace(/\/+$/, "");
  const pathname = path.join("/");
  return `${base}/${pathname}${request.nextUrl.search}`;
}

async function proxy(request: NextRequest, context: RouteContext) {
  const url = resolveBackendUrl(request, context.params.path ?? []);
  const headers = new Headers();
  for (const key of ["content-type", "authorization", "x-request-id", "x-trace-id"]) {
    const value = request.headers.get(key);
    if (value) headers.set(key, value);
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : await request.text(),
      cache: "no-store",
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        code: "MODEL_OPS_PROXY_ERROR",
        message: `ModelOps API 连接失败：${url}`,
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 502 },
    );
  }

  const responseHeaders = new Headers(response.headers);
  responseHeaders.delete("content-encoding");
  responseHeaders.delete("content-length");

  return new NextResponse(await response.arrayBuffer(), {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}

export async function GET(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function PUT(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}
