macro_rules! vk_flags_newtype {
    (
        pub struct $Name:ident($vk_ty:ty);
        default = $default:ident;
        bitor;
        $(const $CONST:ident = $val:expr;)*
    ) => {
        vk_flags_newtype! {
            pub struct $Name($vk_ty);
            default = $default;
            $(const $CONST = $val;)*
        }

        impl std::ops::BitOr for $Name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
    };

    (
        pub struct $Name:ident($vk_ty:ty);
        default = $default:ident;
        $(const $CONST:ident = $val:expr;)*
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $Name(pub(crate) $vk_ty);

        impl $Name {
            $(pub const $CONST: Self = Self($val);)*
        }

        impl Default for $Name {
            fn default() -> Self {
                Self::$default
            }
        }

        impl From<$Name> for $vk_ty {
            fn from(v: $Name) -> Self {
                v.0
            }
        }

        impl From<$vk_ty> for $Name {
            fn from(v: $vk_ty) -> Self {
                Self(v)
            }
        }
    };
}

pub(crate) use vk_flags_newtype;
