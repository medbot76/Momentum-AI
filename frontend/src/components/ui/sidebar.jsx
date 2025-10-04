import * as React from "react"
import { cn } from "../../lib/utils"

const Sidebar = React.forwardRef(({ className, children, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "flex h-screen w-64 flex-col border-r bg-background",
      className
    )}
    {...props}
  >
    {children}
  </div>
))
Sidebar.displayName = "Sidebar"

const SidebarHeader = React.forwardRef(({ className, children, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex h-14 items-center border-b px-4", className)}
    {...props}
  >
    {children}
  </div>
))
SidebarHeader.displayName = "SidebarHeader"

const SidebarContent = React.forwardRef(({ className, children, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex-1 overflow-auto py-2", className)}
    {...props}
  >
    {children}
  </div>
))
SidebarContent.displayName = "SidebarContent"

const SidebarFooter = React.forwardRef(({ className, children, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex h-14 items-center border-t px-4", className)}
    {...props}
  >
    {children}
  </div>
))
SidebarFooter.displayName = "SidebarFooter"

const SidebarNav = React.forwardRef(({ className, children, ...props }, ref) => (
  <nav
    ref={ref}
    className={cn("flex flex-col space-y-1 px-2", className)}
    {...props}
  >
    {children}
  </nav>
))
SidebarNav.displayName = "SidebarNav"

const SidebarNavItem = React.forwardRef(({ className, children, active, ...props }, ref) => (
  <a
    ref={ref}
    className={cn(
      "flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors",
      active
        ? "bg-accent text-accent-foreground"
        : "hover:bg-accent hover:text-accent-foreground",
      className
    )}
    {...props}
  >
    {children}
  </a>
))
SidebarNavItem.displayName = "SidebarNavItem"

export {
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarNav,
  SidebarNavItem,
} 