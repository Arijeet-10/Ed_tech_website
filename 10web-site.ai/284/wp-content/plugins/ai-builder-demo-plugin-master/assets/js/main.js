let guest = false;
let twbb_demo_ab = 'a';
let hubspotSuccessInterval;
let hubspotIntervalActionsCount = 0;

jQuery( document ).keyup( function ( e ) {
  if ( e.key == 'Escape') {
    tbdemo_hide_editor();
  }
} );

// keep this part commented, may be useful later
// jQuery( window ).on( "load", function() {
  // change_cart_page_url();
  // setTimeout(change_cart_page_url, 2000);
  //
  // jQuery( document.body ).on( 'updated_cart_totals', function(){
  //   change_cart_page_url();
  // });
  //
  // jQuery('.twbb_menu-cart__toggle_button').on('click', function (){
  //   change_cart_page_url();
  // });
// });

function change_cart_page_url() {
  if ( typeof taa.home_permalink !== "undefined" && taa.home_permalink != "" ) {
    jQuery(".elementor-button--view-cart").each(function (index) {
      jQuery(this).attr("href", taa.home_permalink + 'cart/');
    });
    jQuery(".elementor-button--checkout, .checkout-button").each(function (index) {
      jQuery(this).attr("href", taa.home_permalink + 'checkout/');
    });
  }
}

jQuery(window).on("load", function() {
  if( jQuery(document).find(".twbb-topbar-domain").length ) {
    jQuery("body").addClass("twbb-topbar-domain-active");
    jQuery("div[data-elementor-type='twbb_header'] .elementor-section-wrap > .elementor-element.elementor-sticky--active").css({
      top: "36px",
    });
  }
});

jQuery(document).ready(function () {
  /* Remove in Elementor edit page preview iframe */
  if ( window.self !== window.top ) {
    jQuery(document).find(".tbdemo-sharebar-container,.tbdemo-ai-popup-layout,.tbdemo-ai-popup-container,.tbdemo-customize-container").remove();
  }

  twbb_demo_ab = tbdemo_getCookie("twbb_demo_ab") ? tbdemo_getCookie("twbb_demo_ab") : 'a';
  if( twbb_demo_ab === 'b' ) {
    jQuery("body").addClass("twbb-b");
    jQuery(document).find(".tbdemo-circle-icon.tbdemo-pro").parent().remove();
    jQuery(document).find(".tbdemo-edit-icon").remove();

  }
  guest = tbdemo_getCookie("tbdemo") ? false : true;
  if(guest){
    jQuery("body").addClass('tbdemo_guest');
  }else{
    jQuery("body").addClass('tbdemo_owner');
    tbdemo_showTypeform();
    if ( window.self === window.top ) {
      let templ = jQuery(document).find("#twbb-topbar-domain-template").html();
      if (templ) {
        jQuery("body").before(templ);
        if (twbb_demo_ab === 'b') {
          jQuery(document).find(".twbb-topbar-domain").addClass("twbb-topbar-domain-ab");
        }
        jQuery("body").css({"top": "36px", "position": "relative"});
      }
    }

  }

  tbdemo_change_domain_id_url();

  jQuery(".tbdemo-circle-icon-content .tbdemo-share-button").parent().on("click", function () {
    jQuery(".tbdemo-big-popup, .tbdemo-big-popup-overlay").show();
  });
  jQuery(".tbdemo-big-popup-close, .tbdemo-big-popup-overlay, .tbdemo-button-promo-got-it").on("click", function () {
    jQuery(".tbdemo-big-popup, .tbdemo-big-popup-overlay").hide();
  });

  jQuery(".tbdemo-popup-copyied").on("click", function () {
    jQuery(this).html("Copied");
    let text_to_copy = "";
    let cont = jQuery(this).parent().find(".tbdemo-popup-text-to-copy");
    if ( typeof cont.data("text") != "undefined" ) {
      text_to_copy += cont.data("text") + " ";
    }
    text_to_copy += cont.html();
    navigator.clipboard.writeText(text_to_copy);
    setTimeout(tbdemo_resetToCopy, 2000, jQuery(this));
  });

  if (guest) {
    jQuery(".tf-v1-popover").hide();
    jQuery(".tbdemo-guest").show();
    jQuery(".twbb-pu-bottom-bar").hide();
  }
  else {
    jQuery(".tbdemo-owner").show();
    if (taa.is_home == 1) {
      // Auto open "One week free" popup once on scrolled Homepage up to 70%.
      jQuery(window).on("scroll", function () {
        let displayed_1wf = tbdemo_getCookie('tbdemo_1WF_' + taa.home_page_id);
        if ( jQuery(".tbdemo-editor-cont").length !== 1 && !displayed_1wf && displayed_1wf === "" && tbdemo_amountscrolled() > 70) {
          tbdemo_setCookie('tbdemo_1WF_' + taa.home_page_id, true);
          tbdemo_show_upgrade_popup("Page scroll");
          if (tbdemo_getCookie("tbdemo_first_time") != 0) {
            tbdemo_customize_container(false);
          }
        }
      });
    }
  }

  /* Add tooltips to the buttons hover.*/
  jQuery(".tf-v1-popover-button:not(.open .tf-v1-popover-button), .tbdemo-circle-icon:not(.tbdemo-close-circle)").hover(function (e) {
      if( jQuery(".tf-v1-popover.open").length || (jQuery(".tbdemo-customize-container").length && jQuery(e.target).hasClass('tbdemo-customize')) ) {
        return;
      }
      jQuery(this).parent().find(".tbdemo-tooltip").show();
    },
    function () {
      jQuery(this).parent().find(".tbdemo-tooltip").hide();
    });

  jQuery(".tbdemo-circle-icon-border .tbdemo-edit").on("click", function() {
    if( jQuery(document).find(".tbdemo-customize-close:visible").length ){
      jQuery(document).find(".tbdemo-customize-close").trigger("click");
    }
    if( jQuery(document).find(".tbdemo-hubspot-close:visible").length ){
      jQuery(document).find(".tbdemo-hubspot-close").trigger("click");
    }
    if ( jQuery(window).width() < 1025 ) {
      tbdemo_show_upgrade_popup("Mobile edit button click");
    }
    else {
      tbdemo_show_editor();
    }
  });

  jQuery(".tbdemo-pro").on("click", function() {
    if( jQuery(document).find(".tbdemo-customize-close:visible").length ){
      jQuery(document).find(".tbdemo-customize-close").trigger("click");
    }
    if( jQuery(document).find(".tbdemo-hubspot-close:visible").length ){
      jQuery(document).find(".tbdemo-hubspot-close").trigger("click");
    }
    tbdemo_show_upgrade_popup("Edit Your Website button topbar");
  });


  jQuery(document).on("click", ".twbb-ab-topbar-new_page-button, .twbb-ab-topbar-edit_website-button",function() {
    if( jQuery(document).find(".tbdemo-customize-close:visible").length ){
      jQuery(document).find(".tbdemo-customize-close").trigger("click");
    }
    if( jQuery(document).find(".tbdemo-hubspot-close:visible").length ){
      jQuery(document).find(".tbdemo-hubspot-close").trigger("click");
    }
    tbdemo_show_upgrade_popup("Add new page button topbar");
  });

  jQuery(document).on("click", ".twbb-topbar-domain-connect-demo", function() {
    jQuery(".twbb-pu-item-text-domain").addClass("twbb-pu-item-text-gradient");
    tbdemo_show_upgrade_popup("Pro button click");

    if ( typeof dataLayer != "undefined" ) {
      let info = 'Front-end';
      if( jQuery(document).find(".tbdemo-editor-cont").length ) {
        info = 'Fake-editor';
      }
      let eventVariant = 'a';
      if( twbb_demo_ab === 'b' ) {
        eventVariant = 'b';
      }
      dataLayer.push({
        event: '10web-event',
        'eventCategory': 'Free Upgrade Offer',
        'eventAction': 'Locked button click',
        'eventLabel' : 'Connect your domain: ' + info,
        'eventVariant': eventVariant
      });
    }

  });

/*
  if ( !guest ) {
    jQuery(document).find(".tbdemo-regenerate-button").show();

    jQuery(document).on("click", ".tbdemo-regenerate-button", function () {
      if (jQuery(this).find(".tbdemo-regenerate-menu").length) {
        jQuery(this).find(".tbdemo-regenerate-menu").toggle();
      } else {
        jQuery(document).find(".tbdemo-generate-popup-layout").toggle();
        jQuery(document).find(".tbdemo-generate-popup").toggle();
        jQuery(document).find("html, body").addClass("twbb-overflow-hidden");
      }
    });

    jQuery(document).on("click", ".tbdemo-menu-generate", function () {
      jQuery(document).find(".tbdemo-generate-popup-layout").toggle();
      jQuery(document).find(".tbdemo-generate-popup").toggle();
      jQuery(document).find("html, body").addClass("twbb-overflow-hidden");
    });

    jQuery(document).on("click", ".tbdemo-menu-regenerate", function () {
      jQuery(document).find(".tbdemo-generate-popup-layout").toggle();
      jQuery(document).find(".tbdemo-regenerate-popup").toggle();
      jQuery(document).find("html, body").addClass("twbb-overflow-hidden");
    });

    jQuery(document).on("click", ".tbdemo-regenerate-popup-close, .tbdemo-generate-popup-layout", function () {
      jQuery(document).find(".tbdemo-regenerate-popup, .tbdemo-generate-popup, .tbdemo-generate-popup-layout").hide();
      jQuery(document).find("html, body").removeClass("twbb-overflow-hidden");
    });
  }
*/

  jQuery(".tf-v1-popover-button").on('click', function () {
    tbdemo_setCookie('tbdemo_displayed_typeform_' + taa.home_page_id, true);
  });

  jQuery(".twbb-pu-upgrade-layout, .twbb-pu-upgrade-close").on("click", function() {
     jQuery("html").removeAttr("style");
     jQuery(".twbb-pu-item-text-gradient.twbb-pu-item-text-domain").removeClass("twbb-pu-item-text-gradient");
  })

  let window_size = jQuery(window).width();
  if( window_size < 550 ) {
    tbdemo_swipePrice( window_size )
  }

  jQuery(document).find(".tbdemo-popup-share-button").on({
    mouseenter: function () {
      jQuery(this).find(".tbdemo-popup-share-button-tooltip").show();
    },
    mouseleave: function () {
      jQuery(this).find(".tbdemo-popup-share-button-tooltip").hide();
    }
  });

  jQuery(".tbdemo-edit-close").on('click', function () {
    tbdemo_hide_editor();
  });

  jQuery(document).on("click", ".single_add_to_cart_button, .add_to_cart_button, .ajax_add_to_cart", function() {
    if ( typeof dataLayer != "undefined" ) {
      dataLayer.push({
        event: '10web-event',
        'eventCategory': 'AI Builder Demo',
        'eventAction': 'Ecommerce: Add to cart',
        'eventLabel' : '-'
      });
    }
  })

  if( jQuery(document).find(".twbb-hubspot-form-container").length ) {
    if (jQuery(window).width() > 860 && parseInt(taa.is_builder2) && !guest) {
      tbdemo_hubspot_form();
    } else {
      jQuery(".twbb-hubspot-form-container").remove();
      jQuery(".tbdemo-sharebar-container").find(".tbdemo-hubspot").closest(".tbdemo-circle-icon-border").remove();
      jQuery(".tbdemo-sharebar-container").removeClass("tbdemo-hubspot-active");
    }
  }

  if( twbb_demo_ab === 'b' && window.self === window.top ) {
    let templ = jQuery(document).find("#twbb-ab-topbar-template").html();
    jQuery("body").before(templ);

    if( jQuery(document).find(".twbb-topbar-domain").length ) {
      jQuery(document).find("#twbb-ab-topbar").addClass("twbb-domain-bar-active");
    }
  }
});

function tbdemo_upgrade_popup_mobile() {
  let upgrade_container = jQuery(document).find(".twbb-pu-upgrade-container");


  let template = '';
  if( jQuery(document).find(".twbb-pu-ecommerce.twbb-pu-upgrade-container").length ) {
    template = jQuery("#twbb-upgrade-mobile-ecommerce-template").html();
  } else {
    template = jQuery("#twbb-upgrade-mobile-template").html();
  }

  if( !upgrade_container.find(".twbb-upgrade-mobile-content").length ) {
    upgrade_container.append(template);

    /* Add this class to show border in case of mobile and content scrollable */
    let content = jQuery(document).find(".twbb-upgrade-mobile-content");
    if( jQuery(document).find(".twbb-pu-ecommerce.twbb-pu-upgrade-container").length ) {
      if ( parseInt(jQuery(window).height()) < 550 ) {
        content.addClass("twbb-upgrade-mobile-content-border");
      }
    } else {
      if ( parseInt(jQuery(window).height()) < 480 ) {
        content.addClass("twbb-upgrade-mobile-content-border");
      }
    }
  }
}

function tbdemo_hubspot_form() {

  /* Check is form already submitted and hide it */
  let tbdemo_hbsp_submitted = tbdemo_getCookie('tbdemo_hbsp_submitted_' + taa.home_page_id);
  if( tbdemo_hbsp_submitted !== "" ) {
    jQuery(".twbb-hubspot-form-container").remove();
    jQuery(".tbdemo-sharebar-container").find(".tbdemo-hubspot").closest(".tbdemo-circle-icon-border").remove();
    jQuery(".tbdemo-sharebar-container").removeClass("tbdemo-hubspot-active");
    return;
  } else {
    jQuery(".tbdemo-sharebar-container").addClass("tbdemo-hubspot-active");
  }

  jQuery(document).on("click", ".tbdemo-hubspot", function() {
    tbdemo_setCookie('tbdemo_hbsp_' + taa.home_page_id, true);
    let client_email = tbdemo_getCookie('client_email');

    if( client_email !== "" ) {
      const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if( re.test(client_email) ) {
        jQuery(document).find(".twbb-hubspot-form-container .hs-email .hs-input").val(client_email);
      }
    }
    let client_id = tbdemo_getCookie('db-user-id');

    if( client_id !== "" ) {
      jQuery(document).find(".twbb-hubspot-form-container .hs-n10web_id .hs-input").val(client_id);
    }
    jQuery(document).find(".twbb-hubspot-form-container .hs-demo_url .hs-input").val(taa.home_permalink);

    if( jQuery(document).find(".tbdemo-customize-close:visible").length ){
      jQuery(document).find(".tbdemo-customize-close").trigger("click");
    }

    jQuery(document).find(".tbdemo-hubspot").hide();
    jQuery(document).find(".tbdemo-hubspot-close").show();

    jQuery(".twbb-hubspot-form-container .hs_submit .actions input").addClass("twbb-disabled-action");
    jQuery(".twbb-hubspot-form-container").show();

  })

  jQuery(document).on("click", ".tbdemo-hubspot-close", function() {
    jQuery(document).find(".tbdemo-hubspot").show();
    jQuery(document).find(".tbdemo-hubspot-close").hide();
    jQuery(".twbb-hubspot-form-container").hide();
  })

  jQuery(document).on("click", ".twbb-hubspot-form-container .hs_submit .actions input", function(e) {
    if( jQuery(this).hasClass("twbb-disabled-action") ) {
      e.preventDefault();
      return false;
    }
    hubspotSuccessInterval = setInterval(hubspotSuccess, 100);
  })

  jQuery(document).on("click", ".hs-form-checkbox-display", function() {
    if ( jQuery(".hs-form-checkbox-display input:checkbox:checked").length == 0 ) {
      jQuery(".twbb-hubspot-form-container .hs_submit .actions input").addClass("twbb-disabled-action");
    } else {
      jQuery(".twbb-hubspot-form-container .hs_submit .actions input").removeClass("twbb-disabled-action");
    }
  })

  jQuery(document).on("click", "li.hs-form-radio label,.twbb-next-button", function(e) {
    jQuery(".twbb-hubspot-form-container .hs_submit .actions input").show();
    jQuery(".twbb-hubspot-form-container .hs_submit .actions .twbb-next-button").hide();

    if( jQuery(this).parent().index() == 0 ) {
      jQuery(".twbb-hubspot-form-container .hs_submit .actions input").removeClass("twbb-disabled-action");
      return;
    }
    jQuery(".twbb-hubspot-form-container .hs-dependent-field > div").first().hide();
    jQuery(".twbb-hubspot-form-container .hs-dependent-field > div").eq(1).show();
    if( !jQuery(document).find(".twbb-hubspot-back-button").length ) {
      jQuery(".twbb-hubspot-form-container .hs_submit .actions").prepend("<div class='twbb-hubspot-back-button'><span>Back</span></div>")
    }

    if ( jQuery(".hs-form-checkbox-display input:checkbox:checked").length == 0 ) {
      jQuery(".twbb-hubspot-form-container .hs_submit .actions input").addClass("twbb-disabled-action");
    } else {
      jQuery(".twbb-hubspot-form-container .hs_submit .actions input").removeClass("twbb-disabled-action");
    }
  })


  jQuery(document).on("click", ".twbb-hubspot-back-button", function() {
    jQuery(document).find(".twbb-hubspot-back-button").remove();
    jQuery(".twbb-hubspot-form-container .hs-dependent-field > div").eq(0).show();
    jQuery(".twbb-hubspot-form-container .hs-dependent-field > div").eq(1).hide();

    if( !jQuery(".twbb-hubspot-form-container .hs_submit .actions twbb-next-button").length ) {
      jQuery(".twbb-hubspot-form-container .hs_submit .actions").append("<span class='twbb-next-button'>Next</span>");
    } else {
      jQuery(".twbb-hubspot-form-container .hs_submit .actions twbb-next-button").show();
    }
    jQuery(".twbb-hubspot-form-container .hs_submit .actions input").hide();
  })

  /* Need to auto open first time */
  let displayed_hbsp = tbdemo_getCookie('tbdemo_hbsp_' + taa.home_page_id);
  if( displayed_hbsp === "") {
    jQuery(window).scroll(function () {
      let scrollTop = jQuery(window).scrollTop();
      let docHeight = jQuery(document).height();
      let windowHeight = jQuery(window).height();
      let scrollPercent = (scrollTop / (docHeight - windowHeight)) * 100;

      if (scrollPercent >= 90 && displayed_hbsp === "") {
        if( tbdemo_getCookie('tbdemo_hbsp_' + taa.home_page_id) !== "" ) {
          return;
        }
        displayed_hbsp = true;
        jQuery(document).find(".tbdemo-hubspot").trigger("click");
      }
    });
  }
}

function hubspotSuccess() {
  hubspotIntervalActionsCount++;
  if( hubspotIntervalActionsCount > 60 ) {
    clearInterval(hubspotSuccessInterval);
    return;
  }
  if( jQuery(document).find(".twbb-hubspot-form-container .submitted-message").length ) {
    jQuery(".twbb-hubspot-form-container").addClass("twbb-hubspot-thankyou");
    setTimeout(function() {
      jQuery(".twbb-hubspot-form-container").remove();
      jQuery(".tbdemo-sharebar-container").find(".tbdemo-hubspot").closest(".tbdemo-circle-icon-border").remove();
      jQuery(".tbdemo-sharebar-container").removeClass("tbdemo-hubspot-active");
      tbdemo_setCookie('tbdemo_hbsp_submitted_' + taa.home_page_id, true);
    },2000);
    clearInterval(hubspotSuccessInterval);
  }
}

function tbdemo_show_editor() {
  jQuery(".tbdemo-edit-close").show();
  let cont = jQuery("<div>", {
    class: "tbdemo-editor-cont",
    onClick: "tbdemo_show_upgrade_popup(\"Left bar click\")"
  });

  jQuery("body").addClass("tbdemo-editor-mode");
  jQuery("body").before(cont);
  jQuery(".tbdemo-editor-cont").before("<div class=\"tbdemo-editor-top-bar\">");

  let dashboard_website_id = (taa.dashboard_website_id) ? (taa.dashboard_website_id) : getCookie('tbdemo_domain_id')
  let add_page_link = twbb.tenweb_dashboard + '/websites/' + dashboard_website_id + '/ai-builder?add_page=1';
  let add_dashboard_link = twbb.tenweb_dashboard + '/websites/' + dashboard_website_id + '/ai-builder?from_demo=1';

  let add_page_btn = jQuery("<a>", {
    class: "tbdemo-add-page",
    html: "<span>+</span>Add Page",
    onClick: "window.open(\"" + add_page_link + "\")"
  });
  let dashboard_btn = jQuery("<a>", {
    class: "tbdemo-dashboard",
    html: "10Web Dashboard",
    onClick: "window.open(\"" + add_dashboard_link + "\")"
  });
  let add_builder_btn = jQuery("<p>", {
    class: "tbdemo-builder-btn",
    html: '10Web Builder<span class="tbdemo-10web-logo"></span><span class="tbdemo-dropdown-icon"></span>'
  });
  jQuery(".tbdemo-editor-top-bar").append('<div class="tbdemo-btn-container"></div>')
  jQuery(".tbdemo-btn-container").append(add_builder_btn);
  jQuery(".tbdemo-btn-container").append(dashboard_btn);

  jQuery(".tbdemo-devicebar-border").hide();
  devicebar(1);

  jQuery(".tbdemo-editor-top-bar").append(add_page_btn);

  jQuery(".tbdemo-editor-cont").append("<div class='tbdemo-editor-basic'>");

  jQuery(".tbdemo-edit").hide();
  //jQuery(".tbdemo-edit").parent().show();
  jQuery("#tbdemo-device-iframe").css({"top": "48px"});

  tbdemo_animate(true);
}

/** Show the fake editor with animation.
 *
 * @param open
 */
function tbdemo_animate(open) {
  jQuery(document).find(".twbb-topbar-domain").remove();
  let sticky_active = false;
  let domainTempl = jQuery(document).find("#twbb-topbar-domain-template").html();
  if(jQuery("div[data-elementor-type='twbb_header'] .elementor-section-wrap > .elementor-element.elementor-sticky--active").length) {
    sticky_active = true;
  }
  let topbarHeightActive = "48px";
  let topbarHeightDeactive = "0";
  if( domainTempl ) {
    topbarHeightActive = "84px";
    topbarHeightDeactive = "36px";
  }

  /* Elementor is adding inline style to section during the resize, reset the styles during the close */
  if( !open ) {
    jQuery("body.tbdemo-editor-mode section").css("left", "0px");
    jQuery(document).find(".twbb-topbar-domain").remove();
  }
  let calc = jQuery(window).width() - 222 + "px";
  jQuery(".tbdemo-editor-top-bar").css({
    height: open ? 0 : "48px"
  }).animate({
    height: open ? "48px" : 0
  }, "slow");
  jQuery("div[data-elementor-type='twbb_header'] .elementor-section-wrap > .elementor-element").css({
    top: open ? 0 : topbarHeightActive,
  }).animate({
    top: (open && sticky_active) ? topbarHeightActive : 0,
    width: open ? (jQuery(window).width()-222)+"px" : jQuery(window).width()+"px",
    left: open ? "unset" : "auto",
  }, "slow");
  jQuery(".tbdemo-editor-cont, .tbdemo-editor-top").css({
    left: open ? "-222px" : 0
  }).animate({
    left: open ? 0 : "-222px"
  }, "slow");

  jQuery("body.tbdemo-editor-mode").css({
    left: open ? 0 : "222px",
    top: open ? topbarHeightDeactive : topbarHeightActive,
    width: open ? "100%" : calc,
  }).animate({
    left: open ? "222px" : 0,
    top: open ? topbarHeightActive : topbarHeightDeactive,
    width: open ? calc : "100%",
  }, "slow", function() {
    if ( open ) {
      tbdemo_fake_editor_marker();
      if( domainTempl ) {
        jQuery("body.tbdemo-editor-mode").css({
          top: "84px",
        });
        jQuery(document).find(".tbdemo-editor-top-bar").append(domainTempl);
        jQuery(document).find(".twbb-topbar-domain").addClass("twbb-topbar-domain-fakeEditor");
        jQuery(document).find(".twbb-topbar-domain.twbb-topbar-domain-fakeEditor").css({width: calc, left: "222px"});
      }

    }
    else {
      tbdemo_hide_editor_complete();
      let left = jQuery(window).width()/2-45;
      jQuery(".tbdemo-devicebar-border").css("left", left);
      jQuery(".tbdemo-devicebar-border").show();
      jQuery("body").css("position", "relative");
      jQuery("body").before(domainTempl);
    }
  });
  /* If iframe mobile is open center the iframe in the new body width */
  if( jQuery("#tbdemo-device-iframe:visible").length ) {
      jQuery(document).find(".tbdemo-device-laptop").removeClass("tbdemo-device-active");
      jQuery(document).find(".tbdemo-device-mobile").addClass("tbdemo-device-active");
      let left = open ? (jQuery(window).width() / 2 + 111 - 180) : (jQuery(window).width() / 2 - 180);

    let buttons_container = jQuery('body .tbdemo-devicebar-border');
    let height_px = parseInt(buttons_container.css("bottom")) + parseInt(buttons_container.css("height")) + 80;

    jQuery("#tbdemo-device-iframe:visible").animate({
        left: left,
        top: 60,
        height: open ? jQuery(window).height()-80 : jQuery(window).height()-height_px
    }, 500);
  } else {
      jQuery("#tbdemo-device-iframe").css({
        width: open ? jQuery("body").width() : jQuery(window).width(),
        left: open ? '222px' : 0
      });
      jQuery(document).find(".tbdemo-device-laptop").addClass("tbdemo-device-active");
      jQuery(document).find(".tbdemo-device-mobile").removeClass("tbdemo-device-active");
  }
}

/**
 *  The function to call on close animation end.
 *  */
function tbdemo_hide_editor_complete() {
  jQuery("body").removeClass("tbdemo-editor-mode");
  jQuery(".tbdemo-editor-cont, .tbdemo-editor-top-bar").remove();
  jQuery(".tbdemo-sharebar-container").show();
  jQuery(".tbdemo_selected_area, .tbdemo_container_edit_settings, .tbdemo_eicon").remove();
  jQuery(".tbdemo-edit").show();
  jQuery(".tbdemo-edit-close").hide();
  let iframe = jQuery('#tbdemo-device-iframe').contents();
  iframe.find(".tbdemo_selected_area, .tbdemo_container_edit_settings, .tbdemo_eicon").remove();
}

function tbdemo_hide_editor() {
  tbdemo_animate(false);
}

function tbdemo_fake_editor_marker() {
  let iframe = jQuery('#tbdemo-device-iframe').contents();
  iframe.find(".tbdemo_selected_area, .tbdemo_container_edit_settings, .tbdemo_eicon").remove();
  jQuery(".tbdemo_selected_area, .tbdemo_container_edit_settings, .tbdemo_eicon").remove();
  let coming_soon_controls = [
    "image",
    "selected_icon",
    "social_icon",
    "selected_active_icon",
    "dismiss_icon",
    "testimonial_image",
    "custom_css",
    "html",
  ]

  let container_edit_icons = "<div class='tbdemo_container_edit tbdemo_selected_area'></div><ul class='tbdemo_eicon tbdemo_container_edit_settings'>";
      container_edit_icons += "<li class='tbdemo_container_edit_settings-add' title='Add Container'>";
      container_edit_icons += "<i class='eicon-plus' aria-hidden='true'></i>";
      container_edit_icons += "</li>";
      container_edit_icons += "<li class='tbdemo_container_edit_settings-edit' title='Edit Container'>";
      container_edit_icons += "<i class='eicon-handle' aria-hidden='true'></i>";
      container_edit_icons += "</li>";
      container_edit_icons += "<li class='tbdemo_container_edit_settings-remove' title='Delete Container'>";
      container_edit_icons += "<i class='eicon-close' aria-hidden='true'></i>";
      container_edit_icons += "</li>";
      container_edit_icons += "</ul>";

  if ( window.self === window.top ) {

    jQuery('body div.elementor[data-elementor-type] > [data-element_type="container"]').each( function() {
      jQuery(this).append(container_edit_icons);
      jQuery(this).find("div[data-element_type='container']").append("<div class='tbdemo_column_edit tbdemo_selected_area'></div><i class='tbdemo_eicon tbdemo_eicon-handle eicon-handle'></i>");
      let widget = jQuery(this).find("div[data-element_type='widget']");

      let ai_template = jQuery("#tbdemo-ai-template").html();
      widget.each( function() {
        let e = jQuery(this);
        e.append("<div class='tbdemo_widget_edit tbdemo_selected_area'></div><i class='tbdemo_eicon tbdemo_eicon-edit eicon-edit'></i>");

        let widget_type =  e.data("widget_type").replace('.default', '');

        if ( ! coming_soon_controls.includes( widget_type ) ) {
          e.find(".tbdemo_selected_area").append(ai_template);
        }
      })


    });
  }

  jQuery(iframe).find('div.elementor[data-elementor-type] > [data-element_type="container"]').each( function() {
    jQuery(this).append(container_edit_icons);
    jQuery(this).find("div[data-element_type='container']").append("<div class='tbdemo_column_edit tbdemo_selected_area'></div><i class='tbdemo_eicon tbdemo_eicon-handle eicon-handle'></i>");
    jQuery(this).find("div[data-element_type='widget']").append("<div class='tbdemo_widget_edit tbdemo_selected_area'></div><i class='tbdemo_eicon tbdemo_eicon-edit eicon-edit'></i>");
    let widget = jQuery(this).find("div[data-element_type='widget']");
    let ai_template = jQuery("#tbdemo-ai-template").html();
    widget.each( function() {
      let e = jQuery(this);
      e.append("<div class='tbdemo_widget_edit tbdemo_selected_area'></div><i class='tbdemo_eicon tbdemo_eicon-edit eicon-edit'></i>");

      let widget_type =  e.data("widget_type").replace('.default', '');
      if ( ! coming_soon_controls.includes( widget_type ) ) {
        e.find(".tbdemo_selected_area").append(ai_template);
      }
    })
  });

  tbdemo_hover_action( jQuery(document) );
  tbdemo_hover_action( iframe );
}

function tbdemo_hover_action( cont ) {
   jQuery(cont).find(".tbdemo_selected_area, .tbdemo_eicon, .tbdemo-ai-front")
     .on( "mouseenter", function(e) {
         e.preventDefault();
         if( jQuery(cont).find(".tbdemo-ai-front-opened").length ) {
           return;
         }
         let div = jQuery( this ).parents().children(".tbdemo_selected_area");
         div.css("opacity", 1);
         div.find(".tbdemo-ai-front").css("opacity", 1);
         div.next(".tbdemo_eicon").css("opacity", 1);
         div.find(".tbdemo-ai-front-button-layer").show();
     })
      .on( "mouseleave", function(e) {
         e.preventDefault();
          if( jQuery(cont).find(".tbdemo-ai-front-opened").length ) {
            return;
          }
          jQuery(cont).find(".tbdemo_selected_area, .tbdemo_eicon:not(.tbdemo_eicon-regenerate)").css("opacity", 0);
      });

      jQuery(cont).find(".tbdemo_selected_area, .tbdemo_eicon").on("click", function(e) {
        if( jQuery(e.target).hasClass('tbdemo-ai-front') || jQuery(e.target).parents('.tbdemo-ai-front').length > 0 ) {
          return;
        }
        if( jQuery(cont).find(".tbdemo-ai-front-opened").length ) {
          jQuery(cont).find(".tbdemo-ai-front-opened").removeClass("tbdemo-ai-front-newpromp-open")
              .removeClass("tbdemo-ai-front-opened")
              .find(".tbdemo-ai-front-button-layer, .tbdemo-ai-front-action-cont, .tbdemo-ai-front-new_prompt-container").hide();
          return;
        }

        tbdemo_show_upgrade_popup("Site content click");
      });

      jQuery(cont).on("click", ".tbdemo-ai-front-button", function(){
        jQuery(this).closest(".tbdemo_selected_area").addClass("tbdemo-ai-front-opened")
        tbdemo_show_ai_action_container(jQuery(this));
      });

      jQuery(cont).on("click", ".tbdemo-ai-front-action-button", function(){
        jQuery(cont).find(".tbdemo-ai-front-opened, .tbdemo-ai-front-newpromp-open")
            .removeClass("tbdemo-ai-front-newpromp-open")
            .removeClass("tbdemo-ai-front-opened")
            .find(".tbdemo-ai-front-button-layer, .tbdemo-ai-front-action-cont, .tbdemo-ai-front-new_prompt-container").hide();
        let eventLabel = jQuery(this).attr("data-eventLabel");
        tbdemo_show_upgrade_popup(eventLabel);
      });

      jQuery(cont).on("click", ".tbdemo-ai-front-new-prompt-button", function(){
        tbdemo_show_new_propmt(jQuery(this), cont);
      });
}

function tbdemo_show_new_propmt(that, cont) {
  if ( typeof dataLayer != "undefined" ) {
    dataLayer.push({
      event: '10web-event',
      'eventCategory': 'Dashboard Action',
      'eventAction': 'AI Builder Demo',
      'eventLabel': 'Write with AI: New prompt'
    });
  }

  let tbdemo_ai_front = jQuery(that).closest(".tbdemo-ai-front");
  tbdemo_ai_front.addClass("tbdemo-ai-front-newpromp-open");
  let templ = jQuery("#tbdemo-ai-new_prompt-template").html();
  if( tbdemo_ai_front.find(".tbdemo-ai-front-new_prompt-container").length ) {
    tbdemo_ai_front.find(".tbdemo-ai-front-new_prompt-container").show();
  } else {
    jQuery(that).parent().after(templ).show();
    tbdemo_ai_front.find(".tbdemo-ai-front-new_prompt-container").show();
  }

  jQuery(this).parent().hide();
  let iframe = jQuery('#tbdemo-device-iframe').contents();
  tbdemo_ai_front_newpropt_click_action_event(iframe);
  tbdemo_ai_front_newpropt_click_action_event(jQuery(document));

  jQuery(cont).find(".tbdemo-ai-front-new_prompt-textarea").on('keyup', function(e) {
    if( jQuery(this).val() == "" ) {
      jQuery(this).parent().find(".tbdemo-ai-front-new_prompt-action-button").addClass("tbdemo-ai-front-button-disabled");
    } else {
      jQuery(this).parent().find(".tbdemo-ai-front-new_prompt-action-button").removeClass("tbdemo-ai-front-button-disabled");
    }
  });
}

function tbdemo_ai_front_newpropt_click_action_event(cont) {

  jQuery(cont).find(".tbdemo-ai-front-new_prompt-action-button").off('click');
  jQuery(cont).find(".tbdemo-ai-front-new_prompt-action-button").on("click", function(){
    if( jQuery(this).hasClass("tbdemo-ai-front-button-disabled") ) {
      return false;
    }
    jQuery(document).find(".tbdemo-ai-front-opened, .tbdemo-ai-front-newpromp-open")
        .removeClass("tbdemo-ai-front-newpromp-open")
        .removeClass("tbdemo-ai-front-opened")
        .find(".tbdemo-ai-front-button-layer, .tbdemo-ai-front-action-cont, .tbdemo-ai-front-new_prompt-container").hide();
    tbdemo_show_upgrade_popup("Generate text");
  });

  tbdemo_ai_front_shift_enter_new_prompt(cont);

}

/* Run new propt action on Enter and new line on Shift+Enter */
function tbdemo_ai_front_shift_enter_new_prompt(cont) {
  jQuery(cont).find('.tbdemo-ai-front-new_prompt-textarea').keydown(function (event) {
    if ( event.key == 'Enter' && !event.shiftKey ) {
      // prevent default behavior
      event.preventDefault();
      jQuery(event.target).parent().find(".tbdemo-ai-front-new_prompt-action-button").trigger("click");
    }
  });
}

function tbdemo_show_ai_action_container(that) {
  if ( typeof dataLayer != "undefined" ) {
    dataLayer.push({
      event: '10web-event',
      'eventCategory': 'Dashboard Action',
      'eventAction': 'AI Builder Demo',
      'eventLabel': 'Write with AI'
    });
  }
  jQuery(that).closest(".tbdemo-ai-front").find(".tbdemo-ai-front-layout, .tbdemo-ai-front-action-cont").show();
}

function tbdemo_show_upgrade_popup(type) {
  tbdemo_upgrade_popup_mobile();
  jQuery(".twbb-pu-upgrade-layout, .twbb-pu-upgrade-container").removeAttr("style").removeClass("twbb-pu-hidden");
  if( !jQuery(document).find(".twbb-pu-ecommerce.twbb-pu-upgrade-container").length ) {
    twbb_pu_run_video(jQuery(".twbb-pu-video-active"));
  }
  jQuery("html").css("overflow", "hidden");

  let eventVariant = 'a';
  if( twbb_demo_ab === 'b' ) {
    eventVariant = 'b';
  }
  if ( typeof dataLayer != "undefined" ) {
    dataLayer.push({
      event: '10web-event',
      'eventCategory': 'Free Upgrade Offer',
      'eventAction': 'Locked button click',
      'eventLabel': 'AI Builder Demo: ' + type,
      'eventVariant': eventVariant
    });
  }
}

function tbdemo_showTypeform(){
  var wrapperElement = document.getElementById('tf-v1-popover');
  let displayed_typeform = tbdemo_getCookie('tbdemo_submitted_typeform_' + taa.home_page_id);
  /* Remove typeform if  user submit once */
  if ( (displayed_typeform && displayed_typeform !== "") || jQuery(window).width() < 860 || typeof window.tf == 'undefined') {
    jQuery(".tbdemo-circle-icon-typeform-cont").remove();
    jQuery(".tbdemo-sharebar-container.tbdemo-typeform-active").removeClass("tbdemo-typeform-active");
    return;
  }


  jQuery(document).find(".tbdemo-customize-container.tbdemo-customize-container-mobile").css("bottom","170px");
  jQuery(".tbdemo-sharebar-container").addClass("tbdemo-typeform-active");

  let params = {
    container: wrapperElement,
    hideHeaders: true,
    hideFooter: true,
    autoClose: 8000,
    height: 400,
    customIcon: "https://images.typeform.com/images/7BGNBNtFKvBc",
    hidden: {
      site_url: window.location.href,
    },
    onSubmit: (event) => {
      tbdemo_setCookie('tbdemo_submitted_typeform_' + taa.home_page_id, true);
    },
    onReady: (event) => {
      // Disable tooltip.
      jQuery(".tbdemo-tooltip").removeClass("tf-v1-popover").hide();
    },
    onClose : (event) => {
      // Enable tooltip.
      jQuery(".tbdemo-tooltip").addClass("tf-v1-popover");
      let displayed_typeform = tbdemo_getCookie('tbdemo_submitted_typeform_' + taa.home_page_id);
      /* Hide typeform if user click on close after submit */
      if ( displayed_typeform && displayed_typeform !== "" ) {
        jQuery(".tbdemo-circle-icon-typeform-cont").remove();
        jQuery(".tbdemo-sharebar-container.tbdemo-typeform-active").removeClass("tbdemo-typeform-active");
      }
      tbdemo_setCookie('tbdemo_displayed_typeform_' + taa.home_page_id, true);
    },
  }

  let displayed_typeform_popup = tbdemo_getCookie('tbdemo_displayed_typeform_' + taa.home_page_id);
  if ( !displayed_typeform_popup ) {
    params['open'] = 'scroll';
    params['openValue'] = 90;
  }
  var formId = "T34BxPCi";
  window.tf.createPopover(
    formId,
    params,
  )}

function tbdemo_change_url_param( url, param, value ) {
  url = new URL(url);
  let search_params = url.searchParams;
  search_params.set(param, value);
  url.search = search_params.toString();
  return url.toString();
}

function tbdemo_swipePrice( window_size ) {
  jQuery(".tbdemo-upgrade-price-content").on("swipeleft", function (e) {
    e.preventDefault();
    let left = jQuery(window).width() - 575;
    jQuery(this).animate({"left": left + "px"});
  });
  jQuery(".tbdemo-upgrade-price-content").on("swiperight", function (e) {
    e.preventDefault();
    jQuery(this).animate({"left": "15px"});
  });
}

function tbdemo_amountscrolled(){
  var winheight= window.innerHeight || (document.documentElement || document.body).clientHeight;
  var docheight = tbdemo_getDocHeight();
  var scrollTop = window.pageYOffset || (document.documentElement || document.body.parentNode || document.body).scrollTop;
  var trackLength = docheight - winheight;
  return Math.floor(scrollTop/trackLength * 100);
}

function tbdemo_getDocHeight() {
  var D = document;
  return Math.max(
    D.body.scrollHeight, D.documentElement.scrollHeight,
    D.body.offsetHeight, D.documentElement.offsetHeight,
    D.body.clientHeight, D.documentElement.clientHeight
  )
}

function tbdemo_change_domain_id_url() {
  if( typeof twbb_sidebar_vars === 'undefined' ) return;
  let str = twbb_sidebar_vars.upgrade_url;


  const regex = /websites\/(\d+)\//gm;

  let m;
  let domain_id = tbdemo_getCookie("tbdemo_domain_id");
  if (!domain_id) {
    return;
  }

  let mathch1 = "";
  let mathch2 = "";
  while ((m = regex.exec(str)) !== null) {
    // This is necessary to avoid infinite loops with zero-width matches
    if (m.index === regex.lastIndex) {
      regex.lastIndex++;
    }
    mathch1 = m[0];
    mathch2 = m[1];
  }

  twbb_sidebar_vars.upgrade_url = str.replace(mathch1, mathch1.replace(mathch2, domain_id));
}

function tbdemo_setCookie(cname, cvalue, exdays) {
  if ( typeof exdays == "undefined" ) {
    var exdays = 3650;
  }
  var d = new Date();
  d.setTime(d.getTime() + (exdays*24*60*60*1000));
  var expires = "expires="+ d.toUTCString();
  document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

function tbdemo_getCookie(cname) {
  let name = cname + "=";
  let decodedCookie = decodeURIComponent(document.cookie);
  let ca = decodedCookie.split(';');
  for(let i = 0; i <ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

function tbdemo_resetToCopy( that ) {
  jQuery(that).html("Copy");
}

/**
 * Run the loader for the given button.
 * @param that
 */
function tbdemo_run_loader(that) {
  if( !that.hasClass("tbdemo-button-disabled") ) {
    return;
  }
  that.addClass("tbdemo-button-loading");
  setTimeout(function () {
    that.removeClass("tbdemo-button-loading tbdemo-button-disabled");
  }, 2000);
}
