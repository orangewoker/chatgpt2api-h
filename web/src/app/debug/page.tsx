"use client";

import { LoaderCircle } from "lucide-react";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAuthGuard } from "@/lib/use-auth-guard";

import { PptPanel } from "./components/ppt-panel";
import { PsdPanel } from "./components/psd-panel";

const tabs = [
  { value: "psd", title: "PSD生成" },
  { value: "ppt", title: "PPT生成" },
];

export default function DebugPage() {
  const { isCheckingAuth, session } = useAuthGuard();

  if (isCheckingAuth || !session) {
    return (
      <div className="flex min-h-[calc(100vh-49px)] items-center justify-center">
        <LoaderCircle className="size-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <Tabs defaultValue="psd" className="mx-auto flex min-h-[calc(100vh-49px)] w-full max-w-[1600px] flex-col gap-4 px-4 pt-3 pb-6 md:px-8">
      <TabsList variant="line" className="w-full">
        {tabs.map(({ value, title }) => (
          <TabsTrigger key={value} value={value}>
            {title}
          </TabsTrigger>
        ))}
      </TabsList>
      <TabsContent value="ppt" className="min-h-0">
        <PptPanel />
      </TabsContent>
      <TabsContent value="psd" className="min-h-0">
        <PsdPanel />
      </TabsContent>
    </Tabs>
  );
}
