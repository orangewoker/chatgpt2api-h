"use client";

import { memo, useEffect, useRef, useState } from "react";
import { Clock3, Download, EyeOff, LoaderCircle, RotateCcw, Sparkles, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ImageConversation, ImageTurnStatus, StoredImage, StoredReferenceImage } from "@/store/image-conversations";

export type ImageLightboxItem = {
  id: string;
  src: string;
  sizeLabel?: string;
  dimensions?: string;
};

type ImageResultsProps = {
  selectedConversation: ImageConversation | null;
  onOpenLightbox: (images: ImageLightboxItem[], index: number) => void;
  onContinueEdit: (conversationId: string, image: StoredImage | StoredReferenceImage) => void;
  onDeletePrompt: (conversationId: string, turnId: string) => void;
  onDeleteResults: (conversationId: string, turnId: string) => void;
  onReuseTurnConfig: (conversationId: string, turnId: string) => void | Promise<void>;
  onRegenerateTurn: (conversationId: string, turnId: string) => void | Promise<void>;
  onRetryImage: (conversationId: string, turnId: string, imageId: string) => void | Promise<void>;
  onTimeoutRetryContinue: (taskId: string) => void | Promise<void>;
  onDismissErrors: (conversationId: string, turnId: string) => void | Promise<void>;
  formatConversationTime: (value: string) => string;
};

// Blob URL 缓存：避免 base64 超长字符串在 DOM 中，改用短小的 blob: URL
const b64BlobUrlCache = new Map<string, string>();

function getStoredImageSrc(image: StoredImage) {
  if (image.b64_json) {
    let url = b64BlobUrlCache.get(image.b64_json);
    if (!url) {
      const binary = atob(image.b64_json);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: "image/png" });
      url = URL.createObjectURL(blob);
      b64BlobUrlCache.set(image.b64_json, url);
    }
    return url;
  }
  return image.url || "";
}

async function downloadStoredImage(image: StoredImage, index: number) {
  let blob: Blob | null = null;
  try {
    if (image.b64_json) {
      const binary = atob(image.b64_json);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      blob = new Blob([bytes], { type: "image/png" });
    } else if (image.url) {
      // 确保 URL 是绝对路径
      const url = image.url.startsWith("http") ? image.url : `${window.location.origin}${image.url}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      blob = await res.blob();
    } else {
      return;
    }
  } catch (err) {
    console.error("Failed to download image:", err);
    // 如果 fetch 失败，尝试直接在新窗口打开
    if (image.url) {
      window.open(image.url, "_blank");
    }
    return;
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `image-${index + 1}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function ImageResults({
  selectedConversation,
  onOpenLightbox,
  onContinueEdit,
  onDeletePrompt,
  onDeleteResults,
  onReuseTurnConfig,
  onRegenerateTurn,
  onRetryImage,
  onTimeoutRetryContinue,
  onDismissErrors,
  formatConversationTime,
}: ImageResultsProps) {
  const [imageDimensions, setImageDimensions] = useState<Record<string, string>>({});
  const [currentTime, setCurrentTime] = useState(Date.now);
  
  // 仅在存在 loading 图片时启动定时器，避免空闲时无谓重渲染
  const hasLoadingImages = selectedConversation?.turns.some(
    (turn) => !turn.resultsDeleted && turn.images.some((image) => image.status === "loading"),
  );
  useEffect(() => {
    if (!hasLoadingImages) return;
    const timer = setInterval(() => {
      setCurrentTime(Date.now());
    }, 500);
    return () => clearInterval(timer);
  }, [hasLoadingImages]);

  const updateImageDimensions = (id: string, width: number, height: number) => {
    const dimensions = formatImageDimensions(width, height);
    setImageDimensions((current) =>
      current[id] === dimensions ? current : { ...current, [id]: dimensions },
    );
  };

  if (!selectedConversation) {
    return (
      <div className="flex h-full min-h-[260px] items-center justify-center text-center sm:min-h-[420px]">
        <div className="w-full max-w-4xl">
          <h1
            className="text-2xl font-semibold tracking-tight text-stone-950 sm:text-3xl md:text-5xl"
            style={{
              fontFamily: '"Palatino Linotype","Book Antiqua","URW Palladio L","Times New Roman",serif',
            }}
          >
            Turn ideas into images
          </h1>
          <p
            className="mx-auto mt-3 max-w-[280px] text-sm italic tracking-[0.01em] text-stone-500 sm:mt-4 sm:max-w-none sm:text-[15px]"
            style={{
              fontFamily: '"Palatino Linotype","Book Antiqua","URW Palladio L","Times New Roman",serif',
            }}
          >
            在同一窗口里保留本地历史与任务状态，并从已有结果图继续发起新的无状态编辑。
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-[980px] flex-col gap-5 sm:gap-8">
      {selectedConversation.turns.map((turn, turnIndex) => {
        const referenceLightboxImages = turn.referenceImages.map((image, index) => ({
          id: `${turn.id}-reference-${index}`,
          src: image.dataUrl,
        }));
        const successfulTurnImages = turn.images.flatMap((image) => {
          const src = image.status === "success" ? getStoredImageSrc(image) : "";
          return src
            ? [
                {
                  id: image.id,
                  src,
                  sizeLabel: image.b64_json ? formatBase64ImageSize(image.b64_json) : undefined,
                  dimensions: imageDimensions[image.id],
                },
              ]
              : [];
        });
        const channelGroups = Array.from(
          turn.images.reduce(
            (groups, image, index) => {
              const model = image.model || turn.model;
              const channel = model.toLowerCase() === "agnes-image-2.1-flash" ? "agnes" : "chatgpt";
              const group = groups.get(channel) || {
                channel,
                model,
                items: [] as Array<{ image: StoredImage; index: number }>,
              };
              group.items.push({ image, index });
              groups.set(channel, group);
              return groups;
            },
            new Map<string, { channel: "chatgpt" | "agnes"; model: string; items: Array<{ image: StoredImage; index: number }> }>(),
          ).values(),
        );
        const isDualChannel = channelGroups.length > 1;

        return (
          <div key={turn.id} className="flex flex-col gap-3 sm:gap-4">
            {!turn.promptDeleted ? (
              <div className="flex justify-end">
                <div className="max-w-[90%] px-1 py-1 text-[14px] leading-6 text-stone-900 sm:max-w-[82%] sm:text-[15px] sm:leading-7">
                  <div className="mb-1.5 flex flex-wrap justify-end gap-2 text-[11px] text-stone-400 sm:mb-2">
                    <span>第 {turnIndex + 1} 轮</span>
                    <span>
                      {turn.mode === "edit" ? "编辑图" : "文生图"}
                    </span>
                    <span>{getTurnStatusLabel(turn.status)}</span>
                    <span>{formatConversationTime(turn.createdAt)}</span>
                  </div>
                  <div className="text-right">{turn.prompt}</div>
                  <div className="mt-2 flex flex-wrap justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => void onReuseTurnConfig(selectedConversation.id, turn.id)}
                      className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2.5 py-1 text-[11px] font-medium text-stone-600 transition hover:bg-stone-200 hover:text-stone-900"
                    >
                      复用配置
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeletePrompt(selectedConversation.id, turn.id)}
                      className="inline-flex size-6 items-center justify-center rounded-full text-stone-300 transition hover:bg-rose-50 hover:text-rose-500"
                      aria-label="删除提示词记录"
                    >
                      <Trash2 className="size-3" />
                    </button>
                  </div>
                </div>
              </div>
            ) : null}

            {!turn.resultsDeleted ? (
              <div className="flex justify-start">
                <div className="w-full p-1">
                  {turn.referenceImages.length > 0 ? (
                    <div className="mb-4 flex flex-col items-end">
                      <div className="mb-3 text-xs font-medium text-stone-500">本轮参考图</div>
                      <div className="flex flex-wrap justify-end gap-3">
                        {turn.referenceImages.map((image, index) => (
                          <div key={`${turn.id}-${image.name}-${index}`} className="flex flex-col items-end gap-2">
                            <button
                              type="button"
                              onClick={() => onOpenLightbox(referenceLightboxImages, index)}
                              className="group relative h-24 w-24 overflow-hidden border border-stone-200/80 bg-stone-100/60 text-left transition hover:border-stone-300"
                              aria-label={`预览参考图 ${image.name || index + 1}`}
                            >
                              <img
                                src={image.dataUrl}
                                alt={image.name || `参考图 ${index + 1}`}
                                className="absolute inset-0 h-full w-full object-cover transition duration-200 group-hover:scale-[1.02]"
                              />
                            </button>
                            <Button
                              variant="outline"
                              size="sm"
                              className="rounded-full border-stone-200 bg-white text-stone-700 hover:bg-stone-50"
                              onClick={() => onContinueEdit(selectedConversation.id, image)}
                            >
                              <Sparkles className="size-4" />
                              加入编辑
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <div className="mb-3 flex flex-wrap items-center gap-1.5 text-[11px] text-stone-500 sm:mb-4 sm:gap-2 sm:text-xs">
                    <span className="rounded-full bg-stone-100 px-3 py-1">{isDualChannel ? `${turn.count} 张/渠道 · 共 ${turn.images.length} 张` : `${turn.count} 张`}</span>
                    <span className="rounded-full bg-stone-100 px-3 py-1">{getTurnStatusLabel(turn.status)}</span>
                    {turn.status === "queued" ? (
                      <span className="rounded-full bg-amber-50 px-3 py-1 text-amber-700">等待当前对话中的前序任务完成</span>
                    ) : null}
                  </div>

                  <div className={cn("grid gap-3", isDualChannel && "md:grid-cols-2")}>
                    {channelGroups.map((group) => {
                      const completedCount = group.items.filter(({ image }) => image.status === "success").length;
                      return (
                        <section key={group.channel} className="min-w-0 overflow-hidden rounded-2xl border border-stone-200/80 bg-white/70 dark:border-white/10 dark:bg-white/[0.03]">
                          {isDualChannel ? (
                            <header className="flex items-center justify-between gap-3 border-b border-stone-100 px-3 py-2.5 dark:border-white/10">
                              <div className="flex min-w-0 items-center gap-2">
                                <span className={cn("inline-flex size-7 shrink-0 items-center justify-center rounded-lg text-xs font-semibold", group.channel === "agnes" ? "bg-violet-100 text-violet-700 dark:bg-violet-400/15 dark:text-violet-200" : "bg-emerald-100 text-emerald-700 dark:bg-emerald-400/15 dark:text-emerald-200")}>
                                  {group.channel === "agnes" ? "A" : "C"}
                                </span>
                                <span className="min-w-0">
                                  <strong className="block truncate text-xs font-semibold text-stone-800 dark:text-stone-100">{group.channel === "agnes" ? "Agnes" : "ChatGPT"}</strong>
                                  <span className="block truncate text-[10px] text-stone-400">{group.model}</span>
                                </span>
                              </div>
                              <span className="shrink-0 text-[10px] text-stone-400">{completedCount}/{group.items.length} 完成</span>
                            </header>
                          ) : null}
                          <div className="grid grid-cols-2 gap-2 p-2 sm:gap-3 sm:p-3">
                            {group.items.map(({ image, index }, groupIndex) => {
                              const imageSrc = image.status === "success" ? getStoredImageSrc(image) : "";
                              if (image.status === "success" && imageSrc) {
                                const currentIndex = successfulTurnImages.findIndex((item) => item.id === image.id);
                                const sizeLabel = image.b64_json ? formatBase64ImageSize(image.b64_json) : "";
                                const dimensions = imageDimensions[image.id];
                                const imageMeta = [sizeLabel, dimensions].filter(Boolean).join(" · ");

                                return (
                                  <div key={image.id} className="min-w-0">
                                    <LazyImage
                                      src={imageSrc}
                                      alt={`${group.channel === "agnes" ? "Agnes" : "ChatGPT"} 结果 ${groupIndex + 1}`}
                                      className="group block aspect-square w-full cursor-zoom-in overflow-hidden rounded-xl"
                                      onLoad={(event) => {
                                        updateImageDimensions(image.id, event.currentTarget.naturalWidth, event.currentTarget.naturalHeight);
                                      }}
                                      onOpen={() => onOpenLightbox(successfulTurnImages, currentIndex)}
                                    />
                                    <div className="flex min-w-0 flex-col gap-1 px-0.5 py-1 text-[10px] sm:flex-row sm:items-center sm:justify-between sm:gap-2 sm:px-1 sm:py-2 sm:text-xs">
                                      <div className="min-w-0 truncate text-stone-500">
                                        <span>结果 {groupIndex + 1}</span>
                                        {image.durationMs != null ? <span className="text-stone-400 sm:ml-2">{formatDuration(image.durationMs)}</span> : null}
                                        {imageMeta ? <span className="block truncate text-stone-400">{imageMeta}</span> : null}
                                      </div>
                                      <div className="flex shrink-0 items-center gap-1.5">
                                        <Button
                                          variant="outline"
                                          size="sm"
                                          className="h-7 w-7 rounded-full border-stone-200 bg-white px-0 text-[10px] text-stone-700 hover:bg-stone-50 sm:h-8 sm:w-fit sm:px-3 sm:text-xs"
                                          onClick={() => onContinueEdit(selectedConversation.id, image)}
                                          aria-label="加入编辑"
                                        >
                                          <Sparkles className="size-3 sm:size-4" />
                                          <span className="hidden sm:inline">加入编辑</span>
                                        </Button>
                                        <Button
                                          variant="outline"
                                          size="sm"
                                          className="h-7 w-7 rounded-full border-stone-200 bg-white px-0 text-[10px] text-stone-700 hover:bg-stone-50 sm:h-8 sm:w-fit sm:px-3 sm:text-xs"
                                          onClick={() => void downloadStoredImage(image, index)}
                                          aria-label="下载"
                                        >
                                          <Download className="size-3 sm:size-4" />
                                          <span className="hidden sm:inline">下载</span>
                                        </Button>
                                      </div>
                                    </div>
                                  </div>
                                );
                              }

                              if (image.status === "error") {
                                const isTimeoutError = image.error?.includes("超时") && image.taskId;
                                return (
                                  <div key={image.id} className="min-w-0">
                                    <div className={cn("flex aspect-square flex-col items-center justify-center gap-1.5 overflow-hidden rounded-xl border border-rose-200 bg-rose-50 px-2 py-2 text-center text-[10px] leading-4 text-rose-600 sm:gap-3 sm:px-4 sm:text-xs", turn.ratio === "16:9" && "aspect-video", turn.ratio === "9:16" && "aspect-[9/16]", turn.ratio === "4:3" && "aspect-[4/3]", turn.ratio === "3:4" && "aspect-[3/4]")}>
                                      <p className="font-medium">图片 {groupIndex + 1}/{group.items.length}</p>
                                      <span className="line-clamp-2">{image.error || "生成失败"}</span>
                                      <div className="flex flex-wrap items-center justify-center gap-1">
                                        {isTimeoutError ? <button type="button" onClick={() => void onTimeoutRetryContinue(image.taskId!)} className="rounded-full bg-emerald-100 px-2 py-1 font-medium text-emerald-600">继续等待</button> : null}
                                        <button type="button" onClick={() => void onRetryImage(selectedConversation.id, turn.id, image.id)} className="rounded-full bg-white px-2 py-1 font-medium text-rose-600">重试</button>
                                      </div>
                                    </div>
                                  </div>
                                );
                              }

                              const imageTaskStatus = image.taskStatus || (turn.status === "queued" ? "queued" : "running");
                              const imageStatusLabel = imageTaskStatus === "queued" ? "排队中" : getProgressLabel(image.progress);
                              const showElapsed = imageTaskStatus === "running" && image.elapsedSecs != null;
                              const elapsedDisplay = showElapsed
                                ? formatElapsed(image.elapsedUpdatedAt != null ? image.elapsedSecs! + (currentTime - image.elapsedUpdatedAt!) / 1000 : image.elapsedSecs!)
                                : null;
                              return (
                                <div key={image.id} className="min-w-0">
                                  <div className={cn("flex aspect-square flex-col items-center justify-center gap-1.5 overflow-hidden rounded-xl border border-stone-200/80 bg-stone-100/80 px-2 py-3 text-center text-stone-500", turn.ratio === "16:9" && "aspect-video", turn.ratio === "9:16" && "aspect-[9/16]", turn.ratio === "4:3" && "aspect-[4/3]", turn.ratio === "3:4" && "aspect-[3/4]")}>
                                    {imageTaskStatus === "queued" ? <Clock3 className="size-4" /> : <LoaderCircle className="size-4 animate-spin" />}
                                    <p className="text-[10px] font-medium leading-4">图片 {groupIndex + 1}/{group.items.length}</p>
                                    <p className="text-[10px] leading-4 text-stone-400">{imageStatusLabel}</p>
                                  </div>
                                  {elapsedDisplay != null ? <div className="px-0.5 py-1 text-[10px] text-stone-400">{elapsedDisplay}</div> : null}
                                </div>
                              );
                            })}
                          </div>
                        </section>
                      );
                    })}
                  </div>

                  {turn.status === "error" && turn.error ? (
                    <div className="mt-4 flex items-center justify-between border-l-2 border-amber-300 bg-amber-50/70 px-4 py-3 text-sm leading-6 text-amber-700">
                      <span>{turn.error}</span>
                      <button
                        type="button"
                        onClick={() => void onDismissErrors(selectedConversation.id, turn.id)}
                        className="ml-3 inline-flex shrink-0 items-center gap-1 rounded-full bg-amber-100 px-2.5 py-1 text-[11px] font-medium text-amber-700 transition hover:bg-amber-200 hover:text-amber-900"
                      >
                        <EyeOff className="size-3" />
                        忽略错误
                      </button>
                    </div>
                  ) : null}

                  <div className="mt-3 flex items-center gap-1.5 text-[11px] sm:mt-4">
                    <button
                      type="button"
                      onClick={() => void onRegenerateTurn(selectedConversation.id, turn.id)}
                      className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2.5 py-1 font-medium text-stone-500 transition hover:bg-stone-200 hover:text-stone-900"
                    >
                      <RotateCcw className="size-3" />
                      全部重新生成
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteResults(selectedConversation.id, turn.id)}
                      className="inline-flex size-6 items-center justify-center rounded-full text-stone-300 transition hover:bg-rose-50 hover:text-rose-500"
                      aria-label="删除生成结果"
                    >
                      <Trash2 className="size-3" />
                    </button>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function getTurnStatusLabel(status: ImageTurnStatus) {
  if (status === "queued") {
    return "排队中";
  }
  if (status === "generating") {
    return "处理中";
  }
  if (status === "success") {
    return "已完成";
  }
  return "失败";
}

const PROGRESS_LABELS: Record<string, string> = {
  getting_account: "确认可用账号",
  uploading: "上传图片",
  bootstrapping: "预热首页",
  getting_token: "获取 token",
  preparing_conversation: "准备会话",
  starting_generation: "启动生成",
  generating: "生成中",
  receiving_image: "接收图片中",
};

function getProgressLabel(progress?: string) {
  if (!progress) {
    return "生成中";
  }
  return PROGRESS_LABELS[progress] || "生成中";
}

function formatElapsed(seconds: number): string {
  return `${seconds.toFixed(1)}s`;
}

function formatDuration(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

const base64SizeCache = new Map<string, string>();
function formatBase64ImageSize(base64: string) {
  let cached = base64SizeCache.get(base64);
  if (cached !== undefined) return cached;
  const normalized = base64.replace(/\s/g, "");
  const padding = normalized.endsWith("==") ? 2 : normalized.endsWith("=") ? 1 : 0;
  const bytes = Math.max(0, Math.floor((normalized.length * 3) / 4) - padding);

  if (bytes >= 1024 * 1024) {
    cached = `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  } else if (bytes >= 1024) {
    cached = `${(bytes / 1024).toFixed(1)} KB`;
  } else {
    cached = `${bytes} B`;
  }
  base64SizeCache.set(base64, cached);
  return cached;
}

function formatImageDimensions(width: number, height: number) {
  return `${width} x ${height}`;
}

const LazyImage = memo(function LazyImage({ src, alt, className, onLoad, onOpen }: {
  src: string;
  alt: string;
  className: string;
  onLoad?: (event: React.SyntheticEvent<HTMLImageElement>) => void;
  onOpen?: () => void;
}) {
  const [isVisible, setIsVisible] = useState(false);
  const imgRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = imgRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "400px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={imgRef} className="relative">
      {isVisible ? (
        <button
          type="button"
          onClick={onOpen}
          className={className}
        >
          <img
            src={src}
            alt={alt}
            className="block h-full w-full object-cover transition duration-200 group-hover:brightness-90 sm:h-auto sm:object-contain"
            onLoad={onLoad}
          />
        </button>
      ) : (
        <div className={`animate-pulse rounded-xl bg-stone-100 min-h-[200px] sm:min-h-[280px] ${className}`} />
      )}
    </div>
  );
});
