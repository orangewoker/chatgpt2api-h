"use client";

import { useMemo, useState } from "react";

import { cn } from "@/lib/utils";

type ImageThumbnailProps = {
  src: string;
  thumbnailSrc?: string;
  alt?: string;
  className?: string;
  imageClassName?: string;
  fallbackToOriginal?: boolean;
};

export function getImageThumbnailUrl(src: string) {
  const marker = "/images/";
  const index = src.indexOf(marker);
  if (index < 0) return src;
  return `${src.slice(0, index)}/image-thumbnails/${src.slice(index + marker.length)}`;
}

export function ImageThumbnail({ src, thumbnailSrc, alt = "", className, imageClassName, fallbackToOriginal = true }: ImageThumbnailProps) {
  const initialSrc = useMemo(() => thumbnailSrc || getImageThumbnailUrl(src), [src, thumbnailSrc]);
  const [loadState, setLoadState] = useState<{ source: string; current: string; failed: boolean }>({
    source: initialSrc,
    current: initialSrc,
    failed: false,
  });
  const isCurrentSource = loadState.source === initialSrc;
  const currentSrc = isCurrentSource ? loadState.current : initialSrc;
  const failed = isCurrentSource && loadState.failed;

  return (
    <span className={cn("block overflow-hidden bg-stone-100", className)}>
      {failed ? (
        <span className="flex h-full min-h-20 w-full items-center justify-center px-3 text-center text-xs text-stone-400">缩略图暂不可用</span>
      ) : (
        <img
          src={currentSrc}
          alt={alt}
          className={cn("h-full w-full object-cover", imageClassName)}
          loading="lazy"
          decoding="async"
          onError={() => {
            if (fallbackToOriginal && currentSrc !== src) {
              setLoadState({ source: initialSrc, current: src, failed: false });
              return;
            }
            setLoadState({ source: initialSrc, current: initialSrc, failed: true });
          }}
        />
      )}
    </span>
  );
}
