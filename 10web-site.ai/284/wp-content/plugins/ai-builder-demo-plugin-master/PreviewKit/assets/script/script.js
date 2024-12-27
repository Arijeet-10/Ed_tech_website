/* Keep in variables upgrade popup Title/Description default texts to bring back after customize upgrade click */
let upgrade_popup_customize_title = '';
let upgrade_popup_customize_descr = '';
/* Open/close Customization popup.*/
function tbdemo_customize_container(show) {
  if ( show && !jQuery("body").hasClass('twbb_old_user')) {

    let template = jQuery("#tbdemo-customize-template").html();
    jQuery("body").addClass("tbdemo-customize-mode");
    if( !jQuery(document).find(".tbdemo-customize-container").length ) {
      jQuery("body").after(template);
    }

    /* For mobile */
    template = template.replace("tbdemo-customize-container", "tbdemo-customize-container tbdemo-customize-container-mobile");
    jQuery("body").append(template);

    if( jQuery(window).width() > 600 ) {
      tbdemo_animate_customization(true);
    } else {
      jQuery(document).find(".tbdemo-customize-container-mobile").show();
    }
    if( jQuery(document).find(".tbdemo-hubspot-close:visible").length ){
      jQuery(document).find(".tbdemo-hubspot-close").trigger("click");
    }

    jQuery(".tbdemo-customize").hide();
    jQuery(".tbdemo-customize-close").show();


    /* Open upgrade popup and change Title/Description */
    jQuery(document).find(".twbb-theme-customize-save").on("click", function() {
      jQuery(document).find(".twbb-pu-upgrade-container").addClass("tbdemo-from-customize");
      upgrade_popup_customize_title = jQuery(document).find(".twbb-pu-upgrade-title").text();
      jQuery(document).find(".twbb-pu-upgrade-title").text(taa.upgrade_popup_customize_title);
      jQuery(document).find(".twbb-pu-upgrade-title").addClass("twbb-customize-title");
      upgrade_popup_customize_descr = jQuery(document).find(".twbb-pu-upgrade-descr").text();
      jQuery(document).find(".twbb-pu-upgrade-descr").text(taa.upgrade_popup_customize_descr);
      tbdemo_show_upgrade_popup("Customize button click");
      tbdemo_customize_container(false);
    });
  }
  else {
    if( jQuery(window).width() > 600 ) {
      tbdemo_animate_customization(false);
    } else {
      jQuery(document).find(".tbdemo-customize-container-mobile").remove();
    }

    jQuery(".tbdemo-customize").show();
    jQuery(".tbdemo-customize-close").hide();
    jQuery(".tbdemo-sharebar-container").css('right', '22px');
  }
  tbdemo_setCookie('tbdemo_first_time', 0);
}

function tbdemo_animate_customization(open) {
  if( open ) {
    jQuery(".tbdemo-sharebar-container").addClass("tbdemo-customize-open");
  } else {
    jQuery(".tbdemo-sharebar-container").removeClass("tbdemo-customize-open");
  }
  jQuery(".tbdemo-customize-container").css({
    right: open ? "-212px" : 0
  }).animate({
    right: open ? 0 : "-212px"
  }, "slow", function (){
    if( !open ) {
      jQuery(".tbdemo-customize-container").remove();
    }
  });
  let calc = jQuery(window).width() - 212 + "px";
  jQuery("body.tbdemo-customize-mode").css({
    right: open ? 0 : "212px",
    left: 0,
    width: open ? "100%" : calc,
  }).animate({
    right: open ? "212px" : 0,
    width: open ? calc : "100%",
  }, "slow");

  jQuery(document).find(".twbb-topbar-domain").animate({
    width: open ? calc : "100%",
  }, "slow");

  if( jQuery("body").hasClass("tbdemo-customize-mode") ) {
    jQuery(document).find("#twbb-ab-topbar").animate({
      width: open ? calc : "100%",
    }, "slow");
  }


  jQuery("div[data-elementor-type='twbb_header'] .elementor-section-wrap > .elementor-element").animate({
    width: open ? (jQuery(window).width()-212)+"px" : jQuery(window).width()+"px"
  }, "slow");

  let left = jQuery(window).width()/2-45;
  if( open ) {
    left = (jQuery(window).width() - 212)/2-45;
  }
  jQuery(".tbdemo-devicebar-border").animate({
    left: left,
  }, 500);


  jQuery(".tbdemo-sharebar-container").css({
    right: open ? "22px" : "222px",
  }).animate({
    right: open ? "222px" : "22px",
  }, "slow", function() {
    if( !open ) {
      jQuery(document).find(".tbdemo-customize-container:not(.tbdemo-customize-container-mobile)").remove();
      jQuery("body").removeClass("tbdemo-customize-mode");
    }
  });

  jQuery(".tbdemo-circle-icon-border .tf-v1-popover-button").css({
    right: open ? "24px" : "246px",
  }).animate({
    right: open ? "246px" : "24px",
  }, "slow");

  let top = 60;
  if( jQuery("body").hasClass("twbb-b") ) {
    if( jQuery(document).find(".twbb-topbar-domain").length ) {
      top = 115;
    } else {
      top = 70;
    }
  }
  /* If iframe mobile is open center the iframe in the new body width */
  if( jQuery("#tbdemo-device-iframe:visible").length ) {
      let left = open ? (jQuery(window).width() / 2 - 111 - 180) : (jQuery(window).width() / 2 - 180);
      jQuery("#tbdemo-device-iframe").animate({
        left: left,
        top: top
      }, 500);
  } else {
      jQuery("#tbdemo-device-iframe").css({
        width: open ? calc : jQuery(window).width(),
        right: open ? '212px' : 0
      });
  }
}

jQuery(document).ready(function () {

  let domainTempl = jQuery(document).find("#twbb-topbar-domain-template").html();
  let topbarHeightActive = "48px";
  if( domainTempl ) {
    topbarHeightActive = "84px";
/*
    jQuery("div[data-elementor-type='twbb_header'] .elementor-section-wrap > .elementor-element.elementor-sticky--active").css({
      top: "36px",
    });
*/
  }
  jQuery(window).on("scroll", function() {
    if( jQuery("body").hasClass("tbdemo-editor-mode") ) {
      jQuery("div[data-elementor-type='twbb_header'] .elementor-section-wrap > .elementor-element.elementor-sticky--active").css({
        top: topbarHeightActive,
      });
      jQuery("div[data-elementor-type='twbb_header'] .elementor-section-wrap > .elementor-element:not(.elementor-sticky--active)").css({
        top: 0,
      });
    }
    else if( domainTempl ) {
        jQuery("div[data-elementor-type='twbb_header'] .elementor-section-wrap > .elementor-element.elementor-sticky--active").css({
          top: "36px",
        });
    }
  })
  jQuery(document).find(".tbdemo-customize-close, .tbdemo-customize-container-overlay").on('click', function () {
    tbdemo_customize_container(false);
  });

  jQuery(".twbb-pu-upgrade-layout, .twbb-pu-upgrade-close").on("click", function() {
    if( jQuery(document).find(".twbb-pu-upgrade-container").hasClass("tbdemo-from-customize") ) {
      jQuery(document).find(".twbb-pu-upgrade-container").removeClass("tbdemo-from-customize");
      jQuery(document).find(".twbb-pu-upgrade-title").text(upgrade_popup_customize_title);
      jQuery(document).find(".twbb-pu-upgrade-descr").text(upgrade_popup_customize_descr);
    }
  })

  jQuery(".tbdemo-customize").on('click', function () {
    tbdemo_customize_container(true);
  });

  if ( tbdemo_getCookie("tbdemo_first_time") != 0 ) {
    tbdemo_customize_container(true);
  }
});

function getCookie(cname) {
  let name = cname + "=";
  let decodedCookie = decodeURIComponent(document.cookie);
  let ca = decodedCookie.split(';');
  for(let i = 0; i < ca.length; i++) {
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